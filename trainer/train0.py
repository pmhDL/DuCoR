import os
import time
from datetime import timedelta
from tqdm import tqdm
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt


def Trainer0(model, train_loader, valid_loader, args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    kwargs_handlers = [kwargs, pg_kwargs]
    if args.iters_to_accumulate > 1:
        accelerator = Accelerator(gradient_accumulation_steps=args.iters_to_accumulate, kwargs_handlers=kwargs_handlers)
    else:
        accelerator = Accelerator(kwargs_handlers=kwargs_handlers)

    device = accelerator.device
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.testst1, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_loader),)
    ## introduce all components to accelerate library
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    valid_loader = accelerator.prepare(valid_loader)

    best_valid_loss = float("inf")
    accelerator.wait_for_everyone()
    auto_loss_curve = []
    auto_loss_curve_val = []

    for epoch in range(args.epochs):
        with tqdm(total=args.batch_size * len(train_loader), disable=not accelerator.is_main_process) as epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}")
            start_time = time.time()
            model.train()
            total_loss = 0.0
            Loss_auto = 0.0
            for i, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        batchloss = model(batch, device)
                    auto_loss = torch.stack(batchloss).mean()
                    loss = auto_loss
                    if args.iters_to_accumulate > 1:
                        loss = loss / accelerator.gradient_accumulation_steps
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                    else:
                        accelerator.backward(loss)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    Loss_auto += auto_loss.item()

                    total_loss += loss.item() * accelerator.gradient_accumulation_steps
                    avg_loss = total_loss / (i + 1)
                    desc = f"Epoch {epoch} - loss {avg_loss:.20f}"
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(batch['images'].size(0))
            auto_loss_curve.append(Loss_auto / (i + 1))

        model.eval()
        with tqdm(total=args.batch_size * len(valid_loader), disable=not accelerator.is_main_process) as epoch_pbar:
            epoch_pbar.set_description(f"VAL Epoch {epoch}")
            total_loss_v = 0.0
            Loss_auto_v = 0.0
            for i, batch in enumerate(valid_loader):
                with torch.no_grad():
                    batchloss = model(batch, device)
                    auto_loss = torch.stack(batchloss).mean()
                    loss = auto_loss
                    Loss_auto_v += auto_loss.item()
                    total_loss_v += loss.item()
                avg_val_loss = total_loss_v / (i + 1)
                desc = f"VAL Epoch {epoch} - loss {avg_val_loss:.20f}"
                epoch_pbar.set_description(desc)
                epoch_pbar.update(batch['images'].size(0))
        auto_loss_curve_val.append(Loss_auto_v / (i + 1))

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            model = accelerator.unwrap_model(model)
            accelerator.save(model.state_dict(), os.path.join(args.out_dir, f"checkpoint_{args.method}.pt"))
        scheduler.step()
        elapsed_time = time.time() - start_time
        print("VAL epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s".format(epoch + 1, args.epochs, avg_loss, avg_val_loss, elapsed_time))

    plot_loss(args, auto_loss_curve, auto_loss_curve_val)

    return model


def plot_loss(args, auto_loss_curve, auto_loss_curve_val):
    fig, ax1 = plt.subplots()
    ax1.plot(list(range(args.epochs)), auto_loss_curve, color='tab:blue', label='train CE loss')
    ax1.plot(list(range(args.epochs)), auto_loss_curve_val, color='tab:red', label='validation CE loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')
    ax1.legend(loc='upper right')
    plt.savefig(os.path.join(args.out_dir, 'auto_regr_loss.png'), bbox_inches='tight', pad_inches=0)
    plt.close()