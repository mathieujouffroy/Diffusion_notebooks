import os
import math
import yaml
import inspect
import argparse
import logging
import numpy as np
import torch, torchvision
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from huggingface_hub import HfFolder, Repository, whoami,  HfApi, create_repo, get_full_repo_name
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


import wandb

from utils import YamlNamespace, set_logging, wandb_cfg

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = logging.getLogger(__name__)


def make_grid(images: list, size: int = 64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


def _extract_into_tensor(arr: np.array, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    """ Parse training paremeters from config YAML file. """

    parser = argparse.ArgumentParser(
        description='Train a diffusion model for image generation.')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="The YAML config file")
    args = parser.parse_args()

    # parse the config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = YamlNamespace(config)

    return config


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args: argparse.Namespace):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    set_logging(args, logging_dir)
    
    augmentations = Compose(
        [
            Resize((args.resolution,args.resolution), interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        logging_dir=logging_dir,
    )

    if args.scheduler == 'ddim':
        scheduler_fct = DDIMScheduler
    else:
        scheduler_fct = DDPMScheduler

    if args.finetune:
        # Prepare pretrained model
        image_pipe = DDPMPipeline.from_pretrained(args.pretrained_model)
        image_pipe.to("cuda")
        # Get a scheduler for sampling
        sampling_scheduler = scheduler_fct.from_config(args.pretrained_model)
        sampling_scheduler.set_timesteps(num_inference_steps=50)
        model = image_pipe.unet
    else:
        model = UNet2DModel(
            sample_size=args.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        accepts_prediction_type = "prediction_type" in set(inspect.signature(scheduler_fct.__init__).parameters.keys())

        if accepts_prediction_type:
            noise_scheduler = scheduler_fct(
                num_train_timesteps=args.ddpm_num_steps,
                beta_schedule=args.ddpm_beta_schedule,
                prediction_type=args.prediction_type,
            )
        else:
            noise_scheduler = scheduler_fct(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_train_steps = args.num_epochs * num_update_steps_per_epoch

    if args.lr_scheduler == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=total_train_steps * args.gradient_accumulation_steps,
        )

        logger.info(f"  LR Scheduler: {lr_scheduler}")
        logger.info(f"  LR Optimizer: {lr_scheduler.optimizer}")
        logger.info(f"  LR Warmup steps before grad acc: {args.lr_warmup_steps}")
        logger.info(f"  LR total steps before grad acc: {total_train_steps}")
        logger.info(f"  LR Warmup steps after grad acc: {args.lr_warmup_steps * args.gradient_accumulation_steps}")
        logger.info(f"  LR total steps after grad acc: {total_train_steps * args.gradient_accumulation_steps}")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    accelerator.register_for_checkpointing(lr_scheduler)

    
    ema_model = EMAModel(
        accelerator.unwrap_model(model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )


    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    if accelerator.is_main_process:
        accelerator.init_trackers(project_name=args.output_dir, config=wandb_cfg(args))
    #if args.logger == 'wandb':
    #    wb_run = set_wandb_project_run(args)

    logger.info("\n\n")
    logger.info(f"  Training set len: {len(train_dataloader)}")
    logger.info(f"  Prediction type: {args.prediction_type}")
    logger.info(f"  Batch Size: {args.train_batch_size}")
    logger.info(f"  Nbr update step per epoch: {num_update_steps_per_epoch}")
    logger.info(f"  Nbr train steps: {total_train_steps}")
    logger.info(f"  Scheduler fct: {scheduler_fct}")
    logger.info(f"  LR Scheduler: {lr_scheduler}")
    logger.info(f"  Log Tracker: {accelerator.trackers}")

    global_step = 0
    for epoch in range(args.num_epochs):
        logger.info(f"  epoch: {epoch}")
        if not args.finetune:
            model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader): #, total=len(train_dataloader)):

            clean_images = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            batch_shape = clean_images.shape[0]

            ##
            # batch.to(accelerator.device) 
            ##

            # Sample a random timestep for each image
            if args.finetune:
                sch_train_steps = image_pipe.scheduler.num_train_timesteps
            else:
                sch_train_steps = noise_scheduler.config.num_train_timesteps

            timesteps = torch.randint(0, sch_train_steps, (batch_shape,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if args.finetune:
                noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)
            else:
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)


            with accelerator.accumulate(model):
                # Predict the noise residual
                if args.finetune:
                    model_output = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]
                else:
                    model_output = model(noisy_images, timesteps).sample

                if args.prediction_type == "epsilon":
                    # Compare the prediction with the actual noise:
                    loss = F.mse_loss(model_output, noise)  # this could have different weights!
                
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")


                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)


                optimizer.step()
                if args.lr_scheduler != 'exponential':
                    lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}

            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                if args.log_images_every == "steps":
                    if (global_step % args.save_images_steps == 0) or (global_step == 1) or (global_step == total_train_steps):
                        if args.finetune:
                            x = torch.randn(8, 3, args.resolution, args.resolution).to('cuda') # Batch of 8
                            for i, t in tqdm(enumerate(sampling_scheduler.timesteps)):
                                model_input = sampling_scheduler.scale_model_input(x, t)
                                with torch.no_grad():
                                    noise_pred = image_pipe.unet(model_input, t)["sample"]
                                x = sampling_scheduler.step(noise_pred, t, x).prev_sample
                            grid = torchvision.utils.make_grid(x, nrow=4)
                            im = grid.permute(1, 2, 0).cpu().clip(-1, 1)*0.5 + 0.5
                            im = Image.fromarray(np.array(im*255).astype(np.uint8))
                            accelerator.log({'Sample generations': wandb.Image(im)}, step=global_step)
                        else:
                            pipeline = DDPMPipeline(
                                unet=accelerator.unwrap_model(ema_model.averaged_model if args.use_ema else model),
                                scheduler=noise_scheduler,
                            )
    
                            generator = torch.Generator(device=pipeline.device).manual_seed(0)
    
                            if args.logger == "tensorboard":
                                # run pipeline in inference (sample random noise and denoise)
                                images = pipeline(
                                    generator=generator,
                                    batch_size=args.eval_batch_size,
                                    output_type="numpy",
                                ).images
        
                                # denormalize the images and save to tensorboard
                                images_processed = (images * 255).round().astype("uint8")
                                accelerator.get_tracker("tensorboard").add_images(
                                    "test_samples", images_processed.transpose(0, 3, 1, 2), global_step
                                )
                            else:
                                images = pipeline(
                                    generator=generator,
                                    batch_size=args.eval_batch_size,
                                ).images
                                im = make_grid(images, args.resolution)
                                accelerator.log({'Sample generations': wandb.Image(im)}, step=global_step)
                
                if args.save_model_every == "steps":
                    if global_step >= 5000:
                        if (global_step % args.save_model_steps == 0) or (global_step == total_train_steps):
                            image_pipe.save_pretrained(f"{args.output_dir}_step_{global_step}")
            
            if global_step >= total_train_steps:
                image_pipe.save_pretrained(f"{args.output_dir}_step_{global_step}")
                break
        
        image_pipe.save_pretrained(f"{args.output_dir}_step_{global_step}")

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if args.log_images_every == "epochs":
                if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(ema_model.averaged_model if args.use_ema else model),
                        scheduler=noise_scheduler,
                    )

                    generator = torch.Generator(device=pipeline.device).manual_seed(0)


                    if args.logger == "tensorboard":
                        # run pipeline in inference (sample random noise and denoise)
                        images = pipeline(
                            generator=generator,
                            batch_size=args.eval_batch_size,
                            output_type="numpy",
                        ).images
                        # denormalize the images and save to tensorboard
                        images_processed = (images * 255).round().astype("uint8")
                        accelerator.get_tracker("tensorboard").add_images(
                            "test_samples", images_processed.transpose(0, 3, 1, 2), epoch
                        )
                    else:
                        images = pipeline(
                            generator=generator,
                            batch_size=args.eval_batch_size,
                        ).images
                        im = make_grid(images, args.resolution)
                        accelerator.log({'Sample generations': wandb.Image(im)})

            if args.save_model_every == "epochs":
                if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                    # save the model
                    pipeline.save_pretrained(f"{args.output_dir}_epoch_{epoch}")
                    if args.push_to_hub:
                        repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)
        
        if args.lr_scheduler == 'exponential':
            lr_scheduler.step()
        
        accelerator.wait_for_everyone()

        

    accelerator.end_training()

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
