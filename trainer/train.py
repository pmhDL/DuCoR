import os
import time
from datetime import timedelta
from tqdm import tqdm
import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from transformers import get_linear_schedule_with_warmup
import torch
from torch.nn import functional as F
from torch.optim import AdamW
import numpy as np
from models.st2.losses import SemiSupLossGMM_M, load_gmm_params, save_gmm_params
from models.st2.reasoning import feat_level_confidence, generate_pseudo_answers, protos_sigma
from utils import plot_loss


def proto_diagnostics(proto_dict, ans_all, update_magnitudes=None):
    if not proto_dict:
        return {"norm_mean": float("nan"),
            "norm_std": float("nan"),
            "update_mean": 0.0,
            "update_std": 0.0,
            "nn_cos_mean": float("nan"),
            "nn_cos_median": float("nan"),
            "coverage": 0.0,
            "answers_total": 0,
            "answers_with_proto": 0,}
    keys = list(proto_dict.keys())
    proto_mat = torch.cat([torch.as_tensor(proto_dict[k]).reshape(1, -1).float() for k in keys], dim=0)
    norms = proto_mat.norm(dim=1).cpu().numpy()
    if proto_mat.size(0) > 1:
        normed = F.normalize(proto_mat, dim=1)
        sim = normed @ normed.t()
        sim.fill_diagonal_(-1.0)
        nn_cos = sim.max(dim=1).values.cpu().numpy()
    else:
        nn_cos = np.asarray([float("nan")], dtype=np.float64)
    unique_answers = set(ans_all)
    answers_with_proto = sum(1 for ans in unique_answers if ans in proto_dict)
    coverage = answers_with_proto / len(unique_answers) if unique_answers else 0.0
    updates = np.asarray(update_magnitudes or [], dtype=np.float64)
    return {"norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "update_mean": float(updates.mean()) if updates.size else 0.0,
        "update_std": float(updates.std()) if updates.size else 0.0,
        "nn_cos_mean": float(np.nanmean(nn_cos)),
        "nn_cos_median": float(np.nanmedian(nn_cos)),
        "coverage": float(coverage),
        "answers_total": len(unique_answers),
        "answers_with_proto": answers_with_proto,}


def selection_ratio(schedule, epoch):
    values = []
    for raw in str(schedule).replace(":", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(min(max(float(raw), 0.0), 1.0))
    if not values:
        return 1.0
    idx = min(max(int(epoch), 0), len(values) - 1)
    return float(values[idx])


def typed_value(args, open_name, close_name, base_value, type_tensor):
    open_value = getattr(args, open_name, None)
    close_value = getattr(args, close_name, None)
    open_value = float(base_value if open_value is None else open_value)
    close_value = float(base_value if close_value is None else close_value)
    values = torch.full_like(type_tensor, open_value, dtype=torch.float32)
    return torch.where(type_tensor == 1, torch.full_like(values, close_value), values)


def find_latest_confidence_cache(cache_dir):
    if not os.path.isdir(cache_dir):
        return None, None, None
    epochs = []
    for name in os.listdir(cache_dir):
        if not name.startswith("gmm_epoch") or not name.endswith(".npz"):
            continue
        raw_epoch = name[len("gmm_epoch"):-len(".npz")]
        if not raw_epoch.isdigit():
            continue
        epoch = int(raw_epoch)
        proto_path = os.path.join(cache_dir, f"protos_sigma_epoch{epoch}.pt")
        gmm_path = os.path.join(cache_dir, name)
        if os.path.exists(proto_path):
            epochs.append((epoch, gmm_path, proto_path))
    if not epochs:
        return None, None, None
    epoch, gmm_path, proto_path = max(epochs, key=lambda item: item[0])
    return epoch, gmm_path, proto_path


def typewise_topk_mask(weights, type_tensor, open_ratio, close_ratio):
    selected = torch.zeros_like(weights, dtype=torch.bool)
    thresholds = []
    for type_value, ratio in ((0, open_ratio), (1, close_ratio)):
        idx = torch.where(type_tensor == type_value)[0]
        total = int(idx.numel())
        if total == 0:
            continue
        k = int(np.ceil(total * ratio))
        k = min(max(k, 0), total)
        if k == 0:
            thresholds.append(0.0)
            continue
        local_weights = weights[idx].detach().float()
        top_vals, top_idx = torch.topk(local_weights, k=k, largest=True, sorted=False)
        selected[idx[top_idx]] = True
        thresholds.append(float(top_vals.min().detach().cpu()))
    threshold = float(np.mean(thresholds)) if thresholds else 0.0
    return selected, threshold


@torch.no_grad()
def semantic_consistency_scores(model, aggembs, answers, device):
    module = model.module if hasattr(model, "module") else model
    if 'gpt2' in module.model_type:
        token_embder = module.text_encoder.transformer.wte
    elif 'phi' in module.model_type or 'stablelm' in module.model_type or 'llama' in module.model_type or 'mistral' in module.model_type:
        if module.args.llmlora == 'lora':
            token_embder = module.text_encoder.base_model.model.model.embed_tokens
        else:
            token_embder = module.text_encoder.model.embed_tokens
    else:
        raise ValueError(f"unsupported model_type for SCR: {module.model_type}")

    tokenizer = module.args.tokenizer
    encoded = tokenizer(
        list(answers),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attn = encoded["attention_mask"].to(device=device, dtype=aggembs.dtype)
    if input_ids.numel() == 0:
        return torch.ones(aggembs.size(0), device=aggembs.device, dtype=aggembs.dtype)

    token_emb = token_embder(input_ids).to(dtype=aggembs.dtype)
    pooled = (token_emb * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1, keepdim=True).clamp(min=1.0)

    if getattr(module.args, "seq_sim", "mean") == "mean" and hasattr(module, "mean_aggregator"):
        answer_feature = module.mean_aggregator(pooled)
    else:
        answer_feature = pooled
    answer_feature = F.normalize(answer_feature, dim=-1)
    joint_feature = F.normalize(aggembs.detach(), dim=-1)
    score = F.cosine_similarity(joint_feature, answer_feature, dim=-1)
    return ((score + 1.0) * 0.5).clamp(min=0.0, max=1.0).to(dtype=aggembs.dtype)


@torch.no_grad()
def evaluate_generation_accuracy(model, val_gen_loader, device, accelerator, args, epoch):
    accelerator.wait_for_everyone()
    model.eval()
    metrics = None
    if accelerator.is_main_process:
        preds_all = []
        refs_all = []
        with torch.inference_mode():
            for batch in val_gen_loader:
                preds = model.generate_bs(batch, device)
                preds_all.extend([str(pred).strip() for pred in preds])
                refs_all.extend([str(ans).strip() for ans in batch["answers"]])

        close_flags = [ans in ("yes", "no") for ans in refs_all]
        correct = [pred == ans for pred, ans in zip(preds_all, refs_all)]
        close_correct = [ok for ok, is_close in zip(correct, close_flags) if is_close]
        open_correct = [ok for ok, is_close in zip(correct, close_flags) if not is_close]

        def _mean_bool(values):
            return float(np.mean(values)) if values else 0.0

        metrics = {
            "acc_all": _mean_bool(correct),
            "count": len(correct),
            "close_count": len(close_correct),
            "open_count": len(open_correct),
        }
        print(f"epoch={epoch} "
            f"acc={metrics['acc_all']:.6f} "
            f"count={metrics['count']} "
            f"close_count={metrics['close_count']} "
            f"open_count={metrics['open_count']}"
        )
    accelerator.wait_for_everyone()
    return metrics


def Trainer(model, train_loader, train_dataset, valid_loader, test_loader, args, val_gen_loader=None):
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
    include_val_in_train = getattr(args, "ducor_include_val_in_train", "no") == "yes"
    save_strategy = getattr(args, "ducor_save_strategy", "val_acc")
    if include_val_in_train and save_strategy != "last":
        raise ValueError("when val is merged into train, use --ducor_save_strategy=last")
    if save_strategy == "val_acc" and not include_val_in_train and val_gen_loader is None:
        raise ValueError("--ducor_save_strategy=val_acc requires a validation generation dataloader")
    run_validation = (not include_val_in_train) and save_strategy in ("val_acc", "val_loss")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_loader),)
    ## introduce all components to accelerate library
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    if run_validation:
        valid_loader = accelerator.prepare(valid_loader)
    else:
        valid_loader = None

    best_valid_loss = float("inf")
    best_val_acc = -1.0
    accelerator.wait_for_everyone()
    auto_loss_curve, contr_loss_curve = [], []
    auto_loss_curve_val = []
    start_time = time.time()
    generate_pseudo_answers(model, test_loader, device, accelerator, args, comp='yes')
    end_time = time.time()
    print(f"running time: {end_time - start_time:.4f} s")

    gmm = SemiSupLossGMM_M(iters=5)
    for epoch in range(args.epochs):
        with tqdm(total=args.batch_size * len(train_loader), disable=not accelerator.is_main_process) as epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}")
            start_time = time.time()
            model.train()
            total_loss = 0.0
            Loss_auto = 0.0
            Loss_contr = 0.0

            epoch_loss = []
            x_or_c = []
            gidx = []
            o_c = []
            ans_lb = []
            gembs = []
            diag_auto, diag_contr, diag_w_loss, diag_w_feat, diag_w_fuse = [], [], [], [], []
            diag_w_loss_values, diag_w_feat_values, diag_w_fuse_values = [], [], []
            diag_w_fuse_open, diag_w_fuse_close = [], []
            diag_conf_agree, diag_alpha_eff = [], []
            diag_agreement_values = []
            diag_scr_scores, diag_scr_reliability = [], []
            cr_missing_count = 0
            cr_total_count = 0
            cr_missing_examples = []
            train_weight_sum = 0.0
            test_weight_sum = 0.0
            train_sample_count = 0
            test_sample_count = 0
            test_nonzero_count = 0
            selection_total_count = 0
            selection_selected_count = 0
            selection_open_total = 0
            selection_open_selected = 0
            selection_close_total = 0
            selection_close_selected = 0
            selection_thresholds = []
            fl_missing_count = 0
            fl_total_count = 0
            fl_missing_examples = []
            confidence_ready = False
            proto_dict = None
            sigma_inv = None
            if epoch > 0:
                prev_epoch = epoch - 1
                gmm_path = os.path.join(args.testst1, f"gmm_epoch{prev_epoch}.npz")
                proto_sig_path = os.path.join(args.testst1, f"protos_sigma_epoch{prev_epoch}.pt")
                missing_paths = []
                if not os.path.exists(gmm_path):
                    missing_paths.append(gmm_path)
                if args.method == 'ducor' and not os.path.exists(proto_sig_path):
                    missing_paths.append(proto_sig_path)

                if missing_paths:
                    if accelerator.is_main_process:
                        print("DUCOR_CONFIDENCE_SKIP "
                            f"epoch={epoch} reason=missing_previous_epoch_cache "
                            f"missing={missing_paths}"
                        )
                else:
                    load_gmm_params(gmm, gmm_path)
                    confidence_ready = True
                    if accelerator.is_main_process:
                        print(f"DUCOR_GMM_LOAD epoch={epoch} path={gmm_path}")
                    if args.method == 'ducor':
                        if accelerator.is_main_process:
                            print(f"DUCOR_PROTO_LOAD epoch={epoch} path={proto_sig_path}")
                        data = torch.load(proto_sig_path, map_location="cpu")
                        proto_dict = {k: v.unsqueeze(0) for k, v in zip(data["keys"], data["proto_mat"])}
                        sigma_inv = data["sigma_inv"]
            else:
                warm_start_mode = getattr(args, "st2_warm_start_confidence", "no")
                warm_epoch, warm_gmm_path, warm_proto_path = find_latest_confidence_cache(args.testst1)
                if warm_start_mode in ("yes", "auto") and warm_gmm_path and warm_proto_path:
                    load_gmm_params(gmm, warm_gmm_path)
                    confidence_ready = True
                    if accelerator.is_main_process:
                        print("DUCOR_WARM_START_LOAD "
                            f"epoch={epoch} source_epoch={warm_epoch} "
                            f"gmm={warm_gmm_path} proto={warm_proto_path}")
                    data = torch.load(warm_proto_path, map_location="cpu")
                    proto_dict = {k: v.unsqueeze(0) for k, v in zip(data["keys"], data["proto_mat"])}
                    sigma_inv = data["sigma_inv"]
                else:
                    if accelerator.is_main_process:
                        print("DUCOR_CONFIDENCE_SKIP epoch=0 reason=no_previous_epoch_cache")
            # ---------------------------------------------------------------------
            if accelerator.is_main_process:
                current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")
                print(f"DUCOR_EPOCH_START epoch={epoch} lr={current_lr:.12g}")

            for i, batch in enumerate(train_loader):
                x_or_c.append(batch['train_or_test'].to('cpu'))
                gidx.append(batch["global_idx"].to('cpu'))
                o_c.append(batch['yn_or_oe'].to('cpu'))
                ans_lb.extend(batch['answers'])
                bs_xc = batch['train_or_test']
                train_mask = (bs_xc == 0)
                test_mask = (bs_xc == 1)

                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        if args.method == 'baseline':
                            batchloss, aggembs = model.forward_st2(batch, device)
                        elif args.method == 'ducor':
                            batchloss, loss_ctrs, aggembs = model.forward_st2(batch, device, proto_dict)
                            cr_stats = getattr(model, "cr_skip_stats", {}) or {}
                            cr_missing_count += int(cr_stats.get("missing", 0))
                            cr_total_count += int(cr_stats.get("total", 0))
                            for example in cr_stats.get("examples", []):
                                if example not in cr_missing_examples and len(cr_missing_examples) < 5:
                                    cr_missing_examples.append(example)
                    # ===============================================
                    bs_loss = torch.stack(batchloss)
                    if args.method == 'ducor':
                        loss_ctrs = torch.stack(loss_ctrs)
                    # ===============================================
                    epoch_loss.extend([ls.item() for ls in batchloss])
                    gembs.append(aggembs)
                    # ========================mask=======================
                    w_ce = torch.zeros_like(bs_loss)
                    if train_mask.any():
                        w_ce[train_mask] = 1.0
                        train_sample_count += int(train_mask.sum().detach().cpu())

                    if test_mask.any():
                        test_oc = batch['yn_or_oe'][test_mask].to(device=bs_loss.device)
                        test_sample_count += int(test_mask.sum().detach().cpu())
                        if not confidence_ready:
                            w_test = torch.zeros(test_mask.sum(), device=bs_loss.device, dtype=bs_loss.dtype)
                            diag_w_fuse.append(float(w_test.detach().float().mean().cpu()))
                        else:
                            ce_te = bs_loss[test_mask].detach().cpu().numpy()
                            oc_te = test_oc.detach().cpu().numpy()
                            w_test_np = gmm.posterior_clean(
                                ce_te,
                                oc_te,
                                posterior_temperature=getattr(args, "st2_ll_posterior_temperature", 1.0),
                                clean_rank_weight=getattr(args, "st2_ll_clean_rank_weight", 0.0),
                                clean_rank_temperature=getattr(args, "st2_ll_clean_rank_temperature", 1.0),
                            )
                            w_test = torch.from_numpy(w_test_np).to(device=bs_loss.device, dtype=bs_loss.dtype)
                            w_loss_tensor = w_test.clone()
                            diag_w_loss.append(float(w_test.detach().float().mean().cpu()))
                            diag_w_loss_values.extend(w_test.detach().float().cpu().numpy().tolist())
                            open_mask = (test_oc == 0)
                            close_mask = (test_oc == 1)
                            if args.method == 'ducor':
                                test_ans = [batch['answers'][i] for i in test_mask.nonzero(as_tuple=True)[0]]
                                feature_temperature = max(float(getattr(args, "st2_feature_temperature", 1.0)), 1e-6)
                                w_feat = feat_level_confidence(aggembs[test_mask], test_ans, proto_dict, sigma_inv, T=feature_temperature,
                                    missing_weight=float(getattr(args, "st2_missing_proto_weight", 1.0)),
                                    answer_types=test_oc.detach().cpu().tolist(),
                                    margin_weight=float(getattr(args, "st2_feature_margin_weight", 0.0)),
                                    margin_temperature=float(getattr(args, "st2_feature_margin_temperature", 0.1)),
                                )
                                fl_stats = getattr(feat_level_confidence, "last_stats", {}) or {}
                                fl_missing_count += int(fl_stats.get("missing", 0))
                                fl_total_count += int(fl_stats.get("total", 0))
                                for example in fl_stats.get("examples", []):
                                    if example not in fl_missing_examples and len(fl_missing_examples) < 5:
                                        fl_missing_examples.append(example)
                                w_feat = torch.as_tensor(w_feat).to(device=bs_loss.device, dtype=bs_loss.dtype)
                                diag_w_feat.append(float(w_feat.detach().float().mean().cpu()))
                                diag_w_feat_values.extend(w_feat.detach().float().cpu().numpy().tolist())
                                alpha_vec = typed_value(args, "st2_open_alpha", "st2_close_alpha", float(args.alpha), test_oc)
                                alpha_vec = alpha_vec.to(device=bs_loss.device, dtype=bs_loss.dtype)
                                alpha_vec = alpha_vec.clamp(min=0.0, max=1.0)
                                agreement_ratio = (torch.minimum(w_test, w_feat) / (torch.maximum(w_test, w_feat) + 1e-6)).clamp(min=0.0, max=1.0)
                                diag_conf_agree.append(float(agreement_ratio.detach().float().mean().cpu()))
                                diag_agreement_values.extend(agreement_ratio.detach().float().cpu().numpy().tolist())
                                diag_alpha_eff.append(float(alpha_vec.detach().float().mean().cpu()))
                                w_test = (w_test ** alpha_vec) * (w_feat ** (1.0 - alpha_vec))
                            open_floor = float(getattr(args, "st2_open_pseudo_weight_floor", 0.0))
                            if open_floor > 0.0:
                                w_test = torch.where(open_mask & (w_test > 0), w_test.clamp(min=open_floor), w_test)
                            close_floor = float(getattr(args, "st2_close_pseudo_weight_floor", 0.0))
                            if close_floor > 0.0:
                                w_test = torch.where(close_mask & (w_test > 0), w_test.clamp(min=close_floor), w_test)
                            if args.method == 'ducor':
                                semantic_scores = semantic_consistency_scores(
                                    model,
                                    aggembs[test_mask].detach(),
                                    test_ans,
                                    device=bs_loss.device,
                                ).to(device=bs_loss.device, dtype=bs_loss.dtype)
                                w_test = (w_test * semantic_scores).clamp(min=0.0, max=1.0)
                                diag_scr_scores.extend(semantic_scores.detach().float().cpu().numpy().tolist())
                                diag_scr_reliability.extend(w_test.detach().float().cpu().numpy().tolist())
                            base_schedule = getattr(args, "st2_selection_schedule", "1.0")
                            open_schedule = getattr(args, "st2_open_selection_schedule", None) or base_schedule
                            close_schedule = getattr(args, "st2_close_selection_schedule", None) or base_schedule
                            open_ratio = selection_ratio(open_schedule, epoch)
                            close_ratio = selection_ratio(close_schedule, epoch)
                            selected, threshold = typewise_topk_mask(w_test, test_oc, open_ratio, close_ratio)
                            w_test = torch.where(selected, w_test, torch.zeros_like(w_test))

                            selection_total_count += int(w_test.numel())
                            selection_selected_count += int(selected.detach().sum().cpu())
                            selection_thresholds.append(threshold)
                            selection_open_total += int(open_mask.detach().sum().cpu())
                            selection_close_total += int(close_mask.detach().sum().cpu())
                            if open_mask.any():
                                selection_open_selected += int((selected & open_mask).detach().sum().cpu())
                            if close_mask.any():
                                selection_close_selected += int((selected & close_mask).detach().sum().cpu())
                            diag_w_fuse.append(float(w_test.detach().float().mean().cpu()))
                            diag_w_fuse_values.extend(w_test.detach().float().cpu().numpy().tolist())
                        open_weights = w_test[test_oc == 0]
                        close_weights = w_test[test_oc == 1]
                        if open_weights.numel() > 0:
                            diag_w_fuse_open.append(float(open_weights.detach().float().mean().cpu()))
                        if close_weights.numel() > 0:
                            diag_w_fuse_close.append(float(close_weights.detach().float().mean().cpu()))
                        test_nonzero_count += int((w_test.detach() > 0).sum().cpu())
                        test_weight_sum += float(w_test.detach().float().sum().cpu())
                        w_ce[test_mask] = w_test

                    train_weight_sum += float(w_ce[train_mask].detach().float().sum().cpu()) if train_mask.any() else 0.0
                    sum_w_ce = w_ce.sum()
                    eps = 1e-12
                    auto_loss = (w_ce * bs_loss).sum() / (sum_w_ce + eps)
                    if args.method == 'ducor':
                        contrastive_loss = (w_ce * loss_ctrs).sum() / (sum_w_ce + eps)
                        effective_cr_weight = float(args.lamda)
                        total_loss = auto_loss + effective_cr_weight * contrastive_loss
                        diag_contr.append(float(contrastive_loss.detach().float().cpu()))
                    else:
                        total_loss = auto_loss
                    diag_auto.append(float(auto_loss.detach().float().cpu()))

                    if args.iters_to_accumulate > 1:
                        total_loss = total_loss / accelerator.gradient_accumulation_steps
                        accelerator.backward(total_loss)
                        if accelerator.sync_gradients:
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                    else:
                        accelerator.backward(total_loss)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    # ======================================
                    Loss_auto += auto_loss.item()
                    if args.method == 'ducor':
                        Loss_contr += contrastive_loss.item()
                    total_loss_scalar = total_loss.item() * accelerator.gradient_accumulation_steps
                    avg_loss = total_loss_scalar / (i + 1)
                    desc = f"Epoch {epoch} - loss {avg_loss:.6f}"
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(batch['images'].size(0))
            auto_loss_curve.append(Loss_auto / (i + 1))
            if args.method == 'ducor':
                contr_loss_curve.append(Loss_contr / (i + 1))

            ## =========================== aggregate loss and features ============================
            gembs = torch.cat(gembs)
            x_or_c = torch.cat(x_or_c)
            o_c = torch.cat(o_c)
            gidx = torch.cat(gidx)
            x_or_c = [int(c.item()) for c in x_or_c]
            o_c = [int(o.item()) for o in o_c]
            gidx = [int(g.item()) for g in gidx]

            epoch_loss_all = accelerate.utils.gather_object(epoch_loss)
            x_or_c_all = accelerate.utils.gather_object(x_or_c)
            o_c_all = accelerate.utils.gather_object(o_c)
            gidx_all = accelerate.utils.gather_object(gidx)
            ans_lb_all = accelerate.utils.gather_object(ans_lb)
            gembs_all = accelerator.gather(gembs)

            gembs_all = gembs_all.detach().to('cpu')
            if accelerator.is_main_process:
                globaldata = {idx: (l, xt, oc, lb, feat) for idx, l, xt, oc, lb, feat in zip(gidx_all, epoch_loss_all, x_or_c_all, o_c_all, ans_lb_all, gembs_all)}
                _, epoch_loss_unique, x_or_c_unique, o_c_unique, ans_lb_unique, gembs_unique = zip(*[(k, v[0], v[1], v[2], v[3], v[4]) for k, v in globaldata.items()])
                epoch_loss_unique, x_or_c_unique, o_c_unique, ans_lb_unique = list(epoch_loss_unique), list(x_or_c_unique), list(o_c_unique), list(ans_lb_unique)
                gembs_unique = torch.stack(gembs_unique)
                print('gembs_unique: ', gembs_unique.size())
                print('epoch_loss_unique: ', len(epoch_loss_unique), 'x_or_c_unique: ', len(x_or_c_unique), 'ans_lb_unique: ', len(ans_lb_unique))
                gmm.fit(epoch_loss_unique, x_or_c_unique, o_c_unique)
                save_gmm_params(gmm, args.testst1, epoch)
                current_proto_dict = protos_sigma(gembs_unique, ans_lb_unique, x_or_c_unique, args.testst1, epoch, eps=1e-6)
                proto_dict = current_proto_dict
                proto_diag = proto_diagnostics(proto_dict, ans_lb_unique, [])


        avg_val_loss = float("nan")
        if run_validation:
            model.eval()
            with tqdm(total=args.batch_size * len(valid_loader), disable=not accelerator.is_main_process) as epoch_pbar:
                epoch_pbar.set_description(f"VAL Epoch {epoch}")
                total_loss_v = 0.0
                Loss_auto_v = 0.0
                val_steps = 0
                for i, batch in enumerate(valid_loader):
                    with torch.no_grad():
                        batchloss = model(batch, device)
                        auto_loss = torch.stack(batchloss).mean()
                        loss = auto_loss

                        Loss_auto_v += auto_loss.item()
                        total_loss_v += loss.item()
                    val_steps = i + 1
                    avg_val_loss = total_loss_v / val_steps
                    desc = f"VAL Epoch {epoch} - loss {avg_val_loss:.20f}"
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(batch['images'].size(0))
            if val_steps > 0:
                auto_loss_curve_val.append(Loss_auto_v / val_steps)

        val_gen_metrics = None
        if save_strategy == "val_acc":
            val_gen_metrics = evaluate_generation_accuracy(model, val_gen_loader, device, accelerator, args, epoch)

        save_checkpoint = False
        save_reason = ""
        if accelerator.is_main_process:
            if save_strategy == "val_acc":
                val_acc = val_gen_metrics["acc_all"]
                if val_acc > best_val_acc:
                    save_checkpoint = True
                    save_reason = "best_val_acc"
                elif val_acc == best_val_acc and avg_val_loss < best_valid_loss:
                    save_checkpoint = True
                    save_reason = "val_acc_tie_lower_val_loss"
                if save_checkpoint:
                    best_val_acc = val_acc
                    best_valid_loss = avg_val_loss
            elif save_strategy == "val_loss":
                if avg_val_loss < best_valid_loss:
                    save_checkpoint = True
                    save_reason = "best_val_loss"
                    best_valid_loss = avg_val_loss
            elif save_strategy == "last":
                save_checkpoint = True
                save_reason = "last"

            if save_checkpoint:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), os.path.join(args.out_dir, f"checkpoint_{args.method}.pt"))
                print("DUCOR_CKPT_SAVE "
                    f"epoch={epoch} reason={save_reason} "
                    f"best_val_acc={best_val_acc:.6f} "
                    f"best_val_loss={best_valid_loss:.6f}")
        accelerator.wait_for_everyone()

        scheduler.step()
        elapsed_time = time.time() - start_time
        if accelerator.is_main_process:
            if run_validation:
                print("VAL epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s".format(epoch + 1, args.epochs, avg_loss, avg_val_loss, elapsed_time))
            else:
                print("TRAIN epoch {}/{} \t loss={:.4f} \t save_strategy={} \t time={:.2f}s".format(epoch + 1, args.epochs, avg_loss, save_strategy, elapsed_time))

        # ================================== update test sample prediction =======================================
        start_time = time.time()
        generate_pseudo_answers(model, test_loader, device, accelerator, args, comp='yes')
        end_time = time.time()
        print(f"running time: {end_time - start_time:.4f} seconds")
        train_dataset.load_test_data()
        # ========================================================================================================
    plot_loss(args, auto_loss_curve, contr_loss_curve, 'train')

    return model