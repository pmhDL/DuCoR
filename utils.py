import collections
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from vit import interpolate_pos_embed


def token_metric(gold_toks, pred_toks):
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, recall


def clean_answer(ans: str) -> str:
    ans = ans.replace('\n', ' ')
    ans = ans.replace('\t', ' ')
    ans = re.sub(r'\s+', ' ', ans)
    return ans.strip()


def load_checkpoint(model, pthpath):
    checkpoint = torch.load(pthpath, map_location='cpu')
    state_dict = checkpoint['model']
    state_dict['pos_embed'] = interpolate_pos_embed(state_dict['pos_embed'], model)
    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % pthpath)
    return model, msg


def plot_loss(args, auto_loss_curve, contr_loss_curve, split):
    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(auto_loss_curve))), auto_loss_curve, color='tab:blue', label='Auto regressive loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Auto regressive loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    has_contrastive = getattr(args, "method", "") == "ducor" and len(contr_loss_curve) > 0
    if has_contrastive:
        ax2 = ax1.twinx()
        ax2.plot(list(range(len(contr_loss_curve))), contr_loss_curve, color='tab:red', label='Contrastive loss')
        ax2.set_ylabel('Contrastive loss', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

    lines, labels = ax1.get_legend_handles_labels()
    if has_contrastive:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='upper right')
    plt.savefig(os.path.join(args.out_dir, split + '_ducor_loss.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
