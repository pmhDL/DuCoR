import re
import os
import json
from dataclasses import dataclass
from typing import Sequence, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
class VQAdataset_test(Dataset):
    def __init__(self, args, path, split='test'):
        super(VQAdataset_test, self).__init__()
        self.IGNORE_INDEX = args.IGNORE_INDEX
        self.IMAGE_TOKEN_INDEX = args.IMAGE_TOKEN_INDEX
        self.args = args
        self.split = split
        imgpath = os.path.join(path, 'images')
        jspath = os.path.join(path, 'json', split + '.json')
        self.jslists = json.load(open(jspath, 'r'))
        for item in self.jslists: item['image_name'] = os.path.join(imgpath, item['image_name'] + '.jpg')

        self.tokenizer = args.tokenizer

        if 'vision' in args.vis_path:
            self.image_processor_train = args.image_processor_train
            self.image_processor_test = args.image_processor_test
        else:
            self.image_processor = args.image_processor

        self.prefix_q = self.tokenizer('question:', return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_ctx = self.tokenizer(' context:', return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_a = self.tokenizer(' answer:', return_tensors="pt", add_special_tokens=False).input_ids[0]

    def __len__(self):
        return len(self.jslists)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        itm = self.jslists[index]
        ans_type = itm['answer_type']
        image_path = itm['image_name']

        image_input = Image.open(image_path).convert('RGB')
        if 'vision' in self.args.vis_path:
            if self.split == 'train':
                image = self.image_processor_train(image_input)
            else:
                image = self.image_processor_test(image_input)
        else:
            image = self.image_processor(image_input, return_tensors="pt").pixel_values[0]

        question = pre_question(itm['question'])
        answer = pre_answer(itm['answer'])
        question_id = self.tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids[0]

        input_id = torch.cat([
            self.prefix_q,
            question_id,
            self.prefix_ctx,
            torch.tensor([self.IMAGE_TOKEN_INDEX], dtype=torch.long),
            self.prefix_a,
        ], dim=0)

        answer_id = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids[0]
        answer_id = torch.cat([self.prefix_a, answer_id], dim=0)

        question_start = len(self.prefix_q)
        question_len = len(question_id)

        data_dict = dict(input_id=input_id, answer_id=answer_id, image=image, question_start=question_start, question_len=question_len,
                         question=question, answer=answer, image_name=image_path, answer_type=ans_type, global_idx=index)
        return data_dict


@dataclass
class VQACollator_test(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_id"] for instance in instances]
        answer_ids = [instance["answer_id"] for instance in instances]
        images = [instance["image"] for instance in instances]
        question_start = [instance["question_start"] for instance in instances]
        question_len = [instance["question_len"] for instance in instances]
        questions = [instance["question"] for instance in instances]
        answers = [instance["answer"] for instance in instances]
        image_names = [instance["image_name"] for instance in instances]
        answer_types = [instance["answer_type"] for instance in instances]
        global_idx = [instance["global_idx"] for instance in instances]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                        batch_first=True,
                                        padding_value=IGNORE_INDEX)
        padded_input_ids = padded_input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = padded_input_ids.ne(IGNORE_INDEX)

        question_start = torch.tensor(question_start, dtype=torch.long)
        question_len = torch.tensor(question_len, dtype=torch.long)
        images = torch.stack(images, dim=0)
        global_idx = torch.tensor(global_idx, dtype=torch.long)

        return {
            "input_ids": padded_input_ids,
            "answer_ids": answer_ids,
            "attention_mask": attention_mask,
            "images": images,
            "question_start": question_start,
            "question_len": question_len,
            "questions": questions,
            "answers": answers,
            "image_names": image_names,
            "answer_types": answer_types,
            "global_idx": global_idx
        }


def pre_question(question):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace(' \t', ' ').replace('is/are', 'is').replace('near/in', 'in')
    question = question.replace('>', 'more than ').replace('-yes/no', '')
    question = question.replace('x ray', 'xray').replace('x-ray', 'xray')
    question = question.rstrip(' ')
    return question


def pre_answer(answer):
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', '-')
    answer = answer.replace('/', ' ')
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer
