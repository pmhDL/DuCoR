import json
import os

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def feat_level_confidence(
    gembs_batch,
    ans_batch,
    proto_dict,
    sigma_inv,
    T=1.0,
    missing_weight=1.0,
    answer_types=None,
    margin_weight=0.0,
    margin_temperature=0.1,
):
    device = gembs_batch.device
    w = torch.empty(gembs_batch.size(0), device=device)
    inv_row = torch.as_tensor(sigma_inv).to(device).reshape(1, -1)
    missing_answers = []
    margin_active = 0
    margin_weight = min(max(float(margin_weight), 0.0), 1.0)
    margin_temperature = max(float(margin_temperature), 1e-6)
    close_keys = [key for key in ("yes", "no") if key in proto_dict]
    open_keys = [key for key in proto_dict.keys() if key not in ("yes", "no")]

    def build_bank(keys):
        if not keys:
            return None
        proto_mat = torch.cat([torch.as_tensor(proto_dict[key]).reshape(1, -1) for key in keys], dim=0)
        return F.normalize(proto_mat.to(device=device, dtype=gembs_batch.dtype), dim=-1)

    close_bank = build_bank(close_keys)
    open_bank = build_bank(open_keys)

    for i, ans in enumerate(ans_batch):
        if ans not in proto_dict:
            missing_answers.append(str(ans))
            w[i] = float(missing_weight)
            continue
        proto = proto_dict[ans]
        diff = gembs_batch[i] - proto.to(device).squeeze(0)
        energy = 0.5 * (diff * diff * inv_row).mean()
        abs_conf = torch.exp(-energy / T).clamp(max=1.0)

        if margin_weight > 0.0:
            is_close = bool(answer_types[i]) if answer_types is not None else ans in ("yes", "no")
            cand_keys = close_keys if is_close else open_keys
            cand_bank = close_bank if is_close else open_bank
            if cand_bank is not None and len(cand_keys) > 1 and ans in cand_keys:
                target_idx = cand_keys.index(ans)
                query = F.normalize(gembs_batch[i].reshape(1, -1), dim=-1)
                sims = (query @ cand_bank.t()).squeeze(0)
                target_sim = sims[target_idx]
                other_mask = torch.ones_like(sims, dtype=torch.bool)
                other_mask[target_idx] = False
                best_other = sims[other_mask].max()
                margin_conf = torch.sigmoid((target_sim - best_other) / margin_temperature)
                abs_conf = (abs_conf ** (1.0 - margin_weight)) * (margin_conf ** margin_weight)
                margin_active += 1

        w[i] = abs_conf.clamp(min=0.0, max=1.0)

    feat_level_confidence.last_stats = {
        "missing": len(missing_answers),
        "total": len(ans_batch),
        "examples": missing_answers[:5],
        "missing_weight": float(missing_weight),
        "margin_active": margin_active,
        "margin_weight": float(margin_weight),
    }
    return w


@torch.no_grad()
def protos_sigma(gembs_all, ans_all, x_or_c_all, save_path, epoch, eps=1e-6):
    os.makedirs(save_path, exist_ok=True)
    feats_cpu = gembs_all.numpy()
    x_or_c_np = np.asarray(x_or_c_all)
    ans_np = np.asarray(ans_all, dtype=object)

    idx_train = np.where(x_or_c_np == 0)[0]
    idx_test = np.where(x_or_c_np == 1)[0]
    ans_train = set(ans_np[idx_train])
    ans_test = set(ans_np[idx_test])

    proto_dict = {}
    for ans in sorted(ans_train | ans_test):
        if ans in ans_train:
            idx = np.where((ans_np == ans) & (x_or_c_np == 0))[0]
        else:
            idx = np.where((ans_np == ans) & (x_or_c_np == 1))[0]
        proto_dict[ans] = feats_cpu[idx].mean(axis=0, keepdims=True)

    resid_sumsq = np.zeros(feats_cpu.shape[1], dtype=np.float32)
    resid_count = 0
    for ans in ans_train:
        idx = np.where((ans_np == ans) & (x_or_c_np == 0))[0]
        diff = feats_cpu[idx] - proto_dict[ans]
        resid_sumsq += np.sum(diff * diff, axis=0)
        resid_count += diff.shape[0]

    sigma_inv = 1.0 / (resid_sumsq / max(resid_count, 1) + eps)
    keys = list(proto_dict.keys())
    proto_mat = torch.cat([torch.from_numpy(proto_dict[k]) for k in keys], dim=0)
    torch.save({"keys": keys, "proto_mat": proto_mat, "sigma_inv": sigma_inv}, os.path.join(save_path, f"protos_sigma_epoch{epoch}.pt"))
    return proto_dict


def generate_pseudo_answers(model, test_dataloader, device, accelerator, args, comp="no"):
    accelerator.wait_for_everyone()
    model.eval()

    if accelerator.is_main_process:
        local_results = []
        with torch.inference_mode():
            for batch in test_dataloader:
                preds = model.generate_bs(batch, device)
                for i, pred in enumerate(preds):
                    local_results.append({
                        "global_idx": int(batch["global_idx"][i]),
                        "image_name": batch["image_names"][i],
                        "answer_type": batch["answer_types"][i],
                        "question": batch["questions"][i],
                        "answer": batch["answers"][i],
                        "predicted": pred,
                    })
                del preds, batch

        out_path = os.path.join(args.testst1, args.testsplit + "_result.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print("local_results: ", len(local_results))
        with open(out_path, "w") as f:
            json.dump(local_results, f)
        if comp == "yes" and getattr(args, "st2_eval_pseudo_metrics", "no") == "yes":
            from test_bs import comput_result

            comput_result(local_results, args.tokenizer)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
