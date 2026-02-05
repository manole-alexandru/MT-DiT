# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import csv
from tqdm import tqdm

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


@torch.no_grad()
def decode_latents(latents, vae):
    """
    Decode VAE latents back to images in [-1, 1].
    """
    return vae.decode(latents / 0.18215).sample


@torch.inference_mode()
def run_validation(
    *,
    ema,
    diffusion,
    vae,
    val_loader,
    num_val_samples,
    latent_size,
    device,
    num_classes,
    logger,
    sample_dir,
    tag,
    cfg_scale=1.5,
    head="blend",
    blend_weight=0.5,
    num_grid_samples=16,
):
    """
    Run validation: collect real images, generate samples, compute FID/IS/KID, and save a sample grid.
    """
    if val_loader is None:
        logger.warning("Validation loader unavailable; skipping validation metrics.")
        return {}

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError as err:
        logger.error(f"torchmetrics not available ({err}); skipping validation metrics.")
        return {}

    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception = InceptionScore(normalize=True, splits=1).to(device)
    kid = KernelInceptionDistance(subset_size=1000, normalize=True).to(device)

    real_seen = 0
    for real_images, _ in tqdm(val_loader, desc=f"Collecting real images ({tag})", leave=False):
        real_images = real_images.to(device)
        fid.update(real_images, real=True)
        kid.update(real_images, real=True)
        real_seen += real_images.size(0)
        if real_seen >= num_val_samples:
            break

    ema.eval()
    head_alias = {"head1": "eps", "head2": "x0", "blend": "blend"}
    gen_head = head_alias.get(head, "blend")
    generated = 0
    grid_samples = []
    grid_count = 0
    batch_size = val_loader.batch_size or 1
    while generated < num_val_samples:
        current_bs = min(batch_size, num_val_samples - generated)
        z = torch.randn(current_bs, ema.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (current_bs,), device=device)
        if cfg_scale > 1.0:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([num_classes] * current_bs, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale, head=gen_head, blend_weight=blend_weight)
            sample_fn = ema.forward_with_cfg
        else:
            model_kwargs = dict(y=y, head=gen_head, blend_weight=blend_weight)
            sample_fn = ema.forward

        samples = diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z.clone(),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)
        samples = decode_latents(samples, vae)
        samples = torch.clamp((samples + 1) / 2, 0, 1)
        if samples.shape[0] > current_bs:
            samples = samples[:current_bs]

        fid.update(samples, real=False)
        kid.update(samples, real=False)
        inception.update(samples)
        generated += samples.shape[0]

        if grid_count < num_grid_samples:
            needed = num_grid_samples - grid_count
            take = samples[:needed].cpu()
            grid_samples.append(take)
            grid_count += take.shape[0]

    if grid_samples:
        grid = torch.cat(grid_samples, dim=0)[:num_grid_samples]
        grid_path = os.path.join(sample_dir, f"{tag}_samples.png")
        save_image(grid, grid_path, nrow=int(num_grid_samples ** 0.5), normalize=True, value_range=(0, 1))
        logger.info(f"Saved validation sample grid to {grid_path}")

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    kid_mean, kid_std = kid.compute()
    metrics = {
        "fid": fid_score,
        "inception_score": is_mean.item(),
        "inception_score_std": is_std.item(),
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
    }
    logger.info(
        f"Metrics ({tag}) -> FID: {metrics['fid']:.4f}, "
        f"IS: {metrics['inception_score']:.4f} +/- {metrics['inception_score_std']:.4f}, "
        f"KID: {metrics['kid_mean']:.6f} +/- {metrics['kid_std']:.6f}"
    )
    return metrics


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        samples_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        csv_path = f"{experiment_dir}/log.csv"
        csv_file = open(csv_path, mode="w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step",
                "epoch",
                "split",
                "loss_total",
                "loss_eps",
                "loss_x0",
                "loss_vb",
                "steps_per_sec",
                "fid",
                "kid_mean",
                "kid_std",
                "inception_score",
                "inception_score_std",
            ],
        )
        csv_writer.writeheader()
    else:
        logger = create_logger(None)
        csv_file, csv_writer = None, None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    eval_diffusion = create_diffusion(str(args.eval_num_steps)) if args.eval_num_steps else diffusion
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    val_loader = None
    train_metrics_loader = None
    if rank == 0:
        val_path = args.val_path or args.data_path
        val_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ])
        val_dataset = ImageFolder(val_path, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Validation set contains {len(val_dataset):,} images ({val_path})")
        train_metrics_dataset = ImageFolder(args.data_path, transform=val_transform)
        train_metrics_loader = DataLoader(
            train_metrics_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Train metrics set contains {len(train_metrics_dataset):,} images ({args.data_path})")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_eps_loss = 0
    running_x0_loss = 0
    running_vb_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        progress = tqdm(loader, disable=rank != 0, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            eps_loss = loss_dict["mse_eps"].mean()
            x0_loss = loss_dict["mse_x0"].mean()
            vb_loss = loss_dict.get("vb", torch.tensor(0.0, device=device)).mean()
            loss = args.loss_weight_eps * eps_loss + args.loss_weight_x0 * x0_loss + vb_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_eps_loss += eps_loss.item()
            running_x0_loss += x0_loss.item()
            running_vb_loss += vb_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_tensor = torch.tensor([
                    running_loss / log_steps,
                    running_eps_loss / log_steps,
                    running_x0_loss / log_steps,
                    running_vb_loss / log_steps
                ], device=device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
                avg_tensor /= dist.get_world_size()
                avg_loss, avg_eps, avg_x0, avg_vb = avg_tensor.tolist()
                logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f} (eps={avg_eps:.4f}, x0={avg_x0:.4f}, vb={avg_vb:.4f}), Steps/Sec: {steps_per_sec:.2f}")
                if rank == 0 and csv_writer is not None:
                    csv_writer.writerow({
                        "step": train_steps,
                        "epoch": epoch,
                        "split": "train",
                        "loss_total": avg_loss,
                        "loss_eps": avg_eps,
                        "loss_x0": avg_x0,
                        "loss_vb": avg_vb,
                        "steps_per_sec": steps_per_sec,
                        "fid": "",
                        "kid_mean": "",
                        "kid_std": "",
                        "inception_score": "",
                        "inception_score_std": "",
                    })
                    csv_file.flush()
                if rank == 0:
                    progress.set_postfix(loss=f"{avg_loss:.4f}", eps=f"{avg_eps:.4f}", x0=f"{avg_x0:.4f}")
                # Reset monitoring variables:
                running_loss = 0
                running_eps_loss = 0
                running_x0_loss = 0
                running_vb_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        if rank == 0:
            epoch_dir = os.path.join(samples_dir, f"epoch_{epoch:04d}")
            os.makedirs(epoch_dir, exist_ok=True)
            eval_heads = ["head1", "head2", "blend"]
            for eval_head in eval_heads:
                head_dir = os.path.join(epoch_dir, eval_head)
                os.makedirs(head_dir, exist_ok=True)
                val_metrics = run_validation(
                    ema=ema,
                    diffusion=eval_diffusion,
                    vae=vae,
                    val_loader=val_loader,
                    num_val_samples=args.num_val_samples,
                    latent_size=latent_size,
                    device=device,
                    num_classes=args.num_classes,
                    logger=logger,
                    sample_dir=head_dir,
                    tag=f"val_{eval_head}",
                    cfg_scale=args.eval_cfg_scale,
                    head=eval_head,
                    blend_weight=args.eval_blend_weight,
                    num_grid_samples=args.num_grid_samples,
                )
                if val_metrics and csv_writer is not None:
                    csv_writer.writerow({
                        "step": train_steps,
                        "epoch": epoch,
                        "split": f"val_{eval_head}",
                        "loss_total": "",
                        "loss_eps": "",
                        "loss_x0": "",
                        "loss_vb": "",
                        "steps_per_sec": "",
                        "fid": val_metrics.get("fid", ""),
                        "kid_mean": val_metrics.get("kid_mean", ""),
                        "kid_std": val_metrics.get("kid_std", ""),
                        "inception_score": val_metrics.get("inception_score", ""),
                        "inception_score_std": val_metrics.get("inception_score_std", ""),
                    })
                    csv_file.flush()
                train_metrics = run_validation(
                    ema=ema,
                    diffusion=eval_diffusion,
                    vae=vae,
                    val_loader=train_metrics_loader,
                    num_val_samples=args.num_val_samples,
                    latent_size=latent_size,
                    device=device,
                    num_classes=args.num_classes,
                    logger=logger,
                    sample_dir=head_dir,
                    tag=f"train_{eval_head}",
                    cfg_scale=args.eval_cfg_scale,
                    head=eval_head,
                    blend_weight=args.eval_blend_weight,
                    num_grid_samples=args.num_grid_samples,
                )
                if train_metrics and csv_writer is not None:
                    csv_writer.writerow({
                        "step": train_steps,
                        "epoch": epoch,
                        "split": f"train_{eval_head}",
                        "loss_total": "",
                        "loss_eps": "",
                        "loss_x0": "",
                        "loss_vb": "",
                        "steps_per_sec": "",
                        "fid": train_metrics.get("fid", ""),
                        "kid_mean": train_metrics.get("kid_mean", ""),
                        "kid_std": train_metrics.get("kid_std", ""),
                        "inception_score": train_metrics.get("inception_score", ""),
                        "inception_score_std": train_metrics.get("inception_score_std", ""),
                    })
                    csv_file.flush()
        dist.barrier()

    logger.info("Done!")
    if rank == 0 and csv_file is not None:
        csv_file.close()
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--loss-weight-eps", type=float, default=1.0)
    parser.add_argument("--loss-weight-x0", type=float, default=1.0)
    parser.add_argument("--val-path", type=str, default=None, help="Optional validation dataset path (ImageFolder). Defaults to data-path.")
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-val-samples", type=int, default=2048)
    parser.add_argument("--num-grid-samples", type=int, default=16)
    parser.add_argument("--eval-cfg-scale", type=float, default=1.5)
    parser.add_argument("--eval-head", type=str, default="blend", help="Generation head for evaluation: head1 (eps), head2 (x0), blend.")
    parser.add_argument("--eval-blend-weight", type=float, default=0.5, help="Blend weight when eval-head=blend.")
    parser.add_argument("--eval-num-steps", type=int, default=250, help="Number of sampling steps for validation generation.")
    args = parser.parse_args()
    main(args)




