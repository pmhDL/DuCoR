import os
import time
import bert_score
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

import json
import torch
import numpy as np
import accelerate
from accelerate import Accelerator
from utils import token_metric


def Tester_BS(model, test_loader, args):
    accelerator = Accelerator()
    device = accelerator.device
    model, test_loader = accelerator.prepare(model, test_loader)
    model = accelerator.unwrap_model(model)

    model.eval()
    local_results = []

    for batch in test_loader:
        with torch.no_grad():
            batch_text = model.generate_bs(batch, device)
        bsize = len(batch_text)
        for i in range(bsize):
            local_results.append({
                "global_idx": int(batch["global_idx"][i]),
                "image_name": batch["image_names"][i],
                "answer_type": batch["answer_types"][i],
                "question": batch["questions"][i],
                "answer": batch["answers"][i],
                "predicted": batch_text[i],
            })
    all_results = accelerate.utils.gather_object(local_results)

    if accelerator.is_main_process:
        uniq = {it["global_idx"]: it for it in all_results}
        all_results_sorted = [uniq[k] for k in sorted(uniq.keys())]

        if args.method == 'baseline':
            f = open(os.path.join(args.testst1, args.testsplit + '_result.json'), 'w')
            json.dump(all_results_sorted, f)
            f.close()
            f = open(os.path.join(args.out_dir, args.testsplit + '_result.json'), 'w')
            json.dump(all_results_sorted, f)
            f.close()
        else:
            f = open(os.path.join(args.out_dir, args.testsplit + '_result_ducor.json'), 'w')
            json.dump(all_results_sorted, f)
            f.close()
        print("predict test samples: ", len(all_results_sorted))
        start_time = time.time()
        comput_result(all_results_sorted, args.tokenizer)
        end_time = time.time()
        print(f"running time: {end_time - start_time:.4f} s")


def comput_result(result_dict, tokenizer):
    refs = [itm['answer'] for itm in result_dict]
    preds = [itm['predicted'] for itm in result_dict]

    is_close = np.array([ans in ('yes', 'no') for ans in refs], dtype=bool)
    is_open  = ~is_close
    # ---------- BERTScore ----------
    _, bert_R, bert_F1 = bert_score.score(refs=refs, cands=preds, model_type='bert-base-uncased', num_layers=12, lang='en', batch_size=32,)

    # ---------- BLEU, token-level, accuracy ----------
    bleu_1_gram = []
    token_f1_list, token_recall_list = [], []
    acc_list, acc_yn_list, acc_oe_list = [], [], []

    for ans, pred in zip(refs, preds):
        ref_tokens = word_tokenize(ans)
        pred_tokens = word_tokenize(pred)
        bleu_1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0))
        bleu_1_gram.append(bleu_1)
        
        # token metric
        gold_tok = tokenizer.encode(ans)
        pred_tok = tokenizer.encode(pred)
        f1_cur, r_cur = token_metric(gold_tok, pred_tok)
        token_f1_list.append(f1_cur)
        token_recall_list.append(r_cur)
        # accuracy
        acc_list.append(pred == ans)
        if ans in ['yes', 'no']:
            acc_yn_list.append(pred == ans)
        else:
            acc_oe_list.append(pred == ans)

    bleu_1_gram = np.array(bleu_1_gram)
    token_f1_list = np.array(token_f1_list)
    token_recall_list = np.array(token_recall_list)

    acc = np.mean(acc_list)
    all_bleu_1_gram = bleu_1_gram.mean()
    
    acc_yn = np.mean(acc_yn_list)
    oe_bleu_1_gram = bleu_1_gram[is_open].mean().item()
    oe_token_f1 = token_f1_list[is_open].mean().item()
    oe_token_recall = token_recall_list[is_open].mean().item()

    print(f"All accuracy={acc:.4f}")
    print(f"All BLEU 1-gram: {all_bleu_1_gram:.4f}")
    print(f"Close accuracy={acc_yn:.4f}")
    print(f"Open BLEU 1-gram: {oe_bleu_1_gram:.4f}")
    print(f"Open F1={oe_token_f1:.4f} Recall={oe_token_recall:.4f}")
