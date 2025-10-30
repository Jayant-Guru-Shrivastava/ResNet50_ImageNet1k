import argparse, os, math, time, yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T
import torchvision.datasets as dset

import timm
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2

from utils import JsonLogger, set_seed

def build_transforms(img_size, aa, reprob):
    train_tf = create_transform(
        input_size=img_size,
        is_training=True,
        auto_augment=aa if aa not in (None, "", "none") else None,
        interpolation='bilinear',
        re_prob=reprob,
        re_mode='pixel',
        re_count=1,
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225),
    )
    eval_tf = create_transform(
        input_size=img_size,
        is_training=False,
        interpolation='bilinear',
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225),
        crop_pct=0.875
    )
    return train_tf, eval_tf

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / target.size(0)))
        return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet50d")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--reprob", type=float, default=0.25)
    ap.add_argument("--aa", type=str, default="rand-m9-mstd0.5")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="runs/r50-imnet")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--dist", action="store_true", help="enable DDP if using torchrun")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    logger = JsonLogger(args.output)

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = args.dist and world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = timm.create_model(args.model, pretrained=False, num_classes=1000)
    model.to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Data
    train_tf, eval_tf = build_transforms(args.img_size, args.aa, args.reprob)
    train_ds = dset.ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
    val_ds   = dset.ImageFolder(os.path.join(args.data, "val"),   transform=eval_tf)

    if is_distributed:
        train_samp = DistributedSampler(train_ds, shuffle=True)
        val_samp   = DistributedSampler(val_ds, shuffle=False)
    else:
        train_samp = None
        val_samp   = None

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                 persistent_workers=True if args.workers>0 else False)
    train_loader = DataLoader(train_ds, shuffle=(train_samp is None), sampler=train_samp, drop_last=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False,                  sampler=val_samp,   drop_last=False, **dl_kw)

    # Criterion & Mixup/Cutmix
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         label_smoothing=args.label_smoothing, num_classes=1000)
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # Optimizer with LR scaled by batch size
    base_bs = 256
    scaled_lr = args.lr * (args.batch_size / base_bs)
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)

    # Cosine LR per-update (with warmup in steps)
    steps_per_epoch = len(train_loader)
    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps - warmup_steps,
        warmup_t=warmup_steps,
        warmup_lr_init=max(1e-6, scaled_lr * 0.01),
        lr_min=1e-6,
        cycle_limit=1,
        t_in_epochs=False,  # per-update schedule
    )

    # EMA
    ema = ModelEmaV2(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
                     decay=0.9999, device=device) if args.ema else None

    scaler = GradScaler(enabled=args.amp)

    # Resume
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(ck["model"])
        else:
            model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if "scaler" in ck and ck["scaler"] is not None and args.amp:
            scaler.load_state_dict(ck["scaler"])
        if ema and "ema" in ck and ck["ema"] is not None:
            ema.module.load_state_dict(ck["ema"])
        start_epoch = int(ck.get("epoch", 0) + 1)
        # place scheduler at correct global step (do not rely on saved state)
        scheduler.step_update(start_epoch * steps_per_epoch)
        if rank == 0:
            print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    def train_one_epoch(epoch):
        if is_distributed and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        if rank==0:
            print(f"Epoch {epoch} training...")
        (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).train()

        total, loss_sum, top1_sum, top5_sum = 0, 0.0, 0.0, 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            if ema: ema.update(model)

            # Cosine step per update
            scheduler.step_update(epoch * steps_per_epoch + step + 1)

            if mixup_fn is None:
                acc1, acc5 = accuracy(logits, y, topk=(1,5))
                top1_sum += acc1.item() * x.size(0) / 100.0
                top5_sum += acc5.item() * x.size(0) / 100.0
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)

        top1 = 100.0 * top1_sum / max(1,total) if mixup_fn is None else None
        top5 = 100.0 * top5_sum / max(1,total) if mixup_fn is None else None
        return loss_sum/total, top1, top5

    @torch.no_grad()
    def evaluate(use_ema=False):
        mdl = ema.module if (ema and use_ema) else (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)
        mdl.eval()
        ce = nn.CrossEntropyLoss()
        total, loss_sum, top1_sum, top5_sum = 0, 0.0, 0.0, 0.0
        for x, y in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(enabled=args.amp):
                logits = mdl(x)
                loss = ce(logits, y)
            acc1, acc5 = accuracy(logits, y, topk=(1,5))
            top1_sum += acc1.item() * x.size(0) / 100.0
            top5_sum += acc5.item() * x.size(0) / 100.0
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
        return loss_sum/total, 100.0*top1_sum/total, 100.0*top5_sum/total

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_top1, tr_top5 = train_one_epoch(epoch)

        val_loss, val_top1, val_top5 = evaluate(use_ema=False)
        ema_loss, ema_top1, ema_top5 = (evaluate(use_ema=True) if ema else (None, None, None))

        dt = time.time() - t0
        if rank==0:
            print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val@model top1={val_top1:.2f} top5={val_top5:.2f}  "
                  f"{'(EMA top1=%.2f top5=%.2f)' % (ema_top1, ema_top5) if ema_top1 is not None else ''}  "
                  f"time={dt:.1f}s")

            # save checkpoint (rank 0 only)
            ck = {
                "epoch": epoch,
                "model": (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "ema": (ema.module.state_dict() if ema else None),
                "args": vars(args),
            }
            out = Path(args.output) / f"epoch{epoch:03d}.pth"
            torch.save(ck, out)

            # log
            JsonLogger(args.output).write({
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "train_top1": float(tr_top1) if tr_top1 is not None else None,
                "train_top5": float(tr_top5) if tr_top5 is not None else None,
                "val_loss": float(val_loss),
                "val_top1": float(val_top1),
                "val_top5": float(val_top5),
                "ema_top1": float(ema_top1) if ema_top1 is not None else None,
                "ema_top5": float(ema_top5) if ema_top5 is not None else None,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "time_sec": float(dt)
            })


    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
