import os
import time
from datetime import timedelta

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler


def warmup_steps(args, total_steps):
    warmup_ratio = float(args.pretrain_warmup_ratio or 0.0)
    if warmup_ratio > 0:
        return int(total_steps * warmup_ratio)
    return int(getattr(args, "warmup_steps", 0) or 0)


def PretrainTrainer(model, train_loader, args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    handlers = [kwargs, pg_kwargs]
    if args.pretrain_gradient_accumulation_steps > 1:
        accelerator = Accelerator(gradient_accumulation_steps=args.pretrain_gradient_accumulation_steps, kwargs_handlers=handlers)
    else:
        accelerator = Accelerator(kwargs_handlers=handlers)

    device = accelerator.device
    os.makedirs(args.out_dir, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_weight_decay)
    total_steps = args.pretrain_epochs * len(train_loader)
    num_warmup_steps = warmup_steps(args, total_steps)
    scheduler = get_scheduler(
        args.pretrain_lr_scheduler_type,
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    if accelerator.is_main_process:
        print("PRETRAIN_SCHEDULER "
            f"steps_per_epoch={len(train_loader)} total_steps={total_steps} "
            f"warmup_steps={num_warmup_steps} warmup_ratio={args.pretrain_warmup_ratio} "
            f"lr={args.pretrain_lr} weight_decay={args.pretrain_weight_decay} scheduler={args.pretrain_lr_scheduler_type}"
        )

    train_curve = []
    save_every = int(getattr(args, "pretrain_save_every", 0) or 0)

    for epoch in range(args.pretrain_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        ce_sum = 0.0
        with tqdm(total=args.pretrain_batch_size * len(train_loader), disable=not accelerator.is_main_process) as pbar:
            pbar.set_description(f"PRETRAIN Epoch {epoch}")
            for i, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        batchloss = model(batch, device)
                        ce_loss = torch.stack(batchloss).mean()
                    loss = ce_loss
                    if args.pretrain_gradient_accumulation_steps > 1:
                        loss = loss / accelerator.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                ce_sum += ce_loss.item()
                total_loss += loss.item() * accelerator.gradient_accumulation_steps
                avg_loss = total_loss / (i + 1)
                pbar.set_description(f"PRETRAIN Epoch {epoch} - loss {avg_loss:.8f}")
                pbar.update(batch["images"].size(0))
        train_loss = ce_sum / max(i + 1, 1)
        train_curve.append(train_loss)

        if accelerator.is_main_process:
            elapsed = time.time() - start_time
            print("PRETRAIN_EPOCH "
                f"epoch={epoch + 1}/{args.pretrain_epochs} train_loss={train_loss:.6f} "
                f"time={elapsed:.2f}s")

        unwrapped = accelerator.unwrap_model(model)
        accelerator.save(unwrapped.state_dict(), os.path.join(args.out_dir, "checkpoint_pretrain.pt"))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            accelerator.save(unwrapped.state_dict(), os.path.join(args.out_dir, f"checkpoint_pretrain_epoch{epoch + 1}.pt"))

    if accelerator.is_main_process:
        np.save(os.path.join(args.out_dir, "pretrain_loss_train.npy"), np.array(train_curve))
    return accelerator.unwrap_model(model)