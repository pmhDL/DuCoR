import os
import json
from tqdm import tqdm
import torch

from test_bs import comput_result


def Tester_SG(model, dataset, args):
    model.eval()
    model = model.to(args.device)

    result = []
    with tqdm(total=len(dataset)) as epoch_pbar:
        epoch_pbar.set_description("Testing")
        for batch in dataset:
            answers_raw = str(batch['answer'])
            with torch.no_grad():
                out_text_all = model.generate(batch, args.device)

            if args.num_beam > 1:
                out_text = out_text_all[0].lower()
            else:
                out_text = out_text_all.lower()

            result_dict = {
                "image_name": batch['image_name'],
                "answer_type": batch['answer_type'],
                "question": batch['question'],
                "answer": answers_raw,
                "predicted": out_text
            }
            result.append(result_dict)
            print('ground truth: ', answers_raw, '    predicted: ', out_text)

    if args.method == 'baseline':
        f = open(os.path.join(args.testst1, args.testsplit + '_result.json'), 'w')
        json.dump(result, f)
        f.close()
        f = open(os.path.join(args.out_dir, args.testsplit + '_result.json'), 'w')
        json.dump(result, f)
        f.close()
    else:
        f = open(os.path.join(args.out_dir, args.testsplit + '_result_ducor.json'), 'w')
        json.dump(result, f)
        f.close()
    print('predict test samples: ', len(result))
    comput_result(result, args.tokenizer)
