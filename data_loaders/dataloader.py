import re
import os
import json
from dataclasses import dataclass
from typing import Sequence, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
class VQAdataset(Dataset):
    def __init__(self, args, path, split='train'):
        super(VQAdataset, self).__init__()
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
        item = self.jslists[index]
        ans_type = item['answer_type']
        image_path = item['image_name']
        image_input = Image.open(image_path).convert('RGB')
        if 'vision' in self.args.vis_path:
            if self.split == 'train':
                image = self.image_processor_train(image_input)
            else:
                image = self.image_processor_test(image_input)
        else:
            image = self.image_processor(image_input, return_tensors="pt").pixel_values[0]

        ans = pre_answer(item['answer'])
        if ans == 'yes' or ans == 'no':
            yn_or_oe = 1
        else:
            yn_or_oe = 0

        question = pre_question(item['question'])
        # answer = pre_answer(item['answer'])
        answer = ans+'<|endoftext|>'
        question_id = self.tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids[0]
        answer_id = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids[0]

        input_id = torch.cat([
            self.prefix_q,
            question_id,
            self.prefix_ctx,
            torch.tensor([self.IMAGE_TOKEN_INDEX], dtype=torch.long),
            self.prefix_a,
            answer_id
        ], dim=0)

        question_start = len(self.prefix_q)
        question_len = len(question_id)
        answer_len = len(answer_id)
        prefix_a_len = len(self.prefix_a)
        answersv = pre_answer(item['answer'])

        data_dict = dict(input_id=input_id, image=image, question_start=question_start, question_len=question_len,
                         prefix_a_len=prefix_a_len, answer_len=answer_len, question=question, answer=answersv,
                         image_name=image_path, answer_type=ans_type, yn_or_oe=yn_or_oe)

        return data_dict


@dataclass
class VQACollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_id"] for instance in instances]
        images = [instance["image"] for instance in instances]
        question_start = [instance["question_start"] for instance in instances]
        question_len = [instance["question_len"] for instance in instances]
        prefix_a_len = [instance["prefix_a_len"] for instance in instances]
        answer_len = [instance["answer_len"] for instance in instances]
        questions = [instance["question"] for instance in instances]
        answers = [instance["answer"] for instance in instances]
        image_names = [instance["image_name"] for instance in instances]
        answer_types = [instance["answer_type"] for instance in instances]
        yn_or_oe = [instance["yn_or_oe"] for instance in instances]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=IGNORE_INDEX)
        padded_input_ids = padded_input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = padded_input_ids.ne(IGNORE_INDEX)

        question_start = torch.tensor(question_start, dtype=torch.long)
        question_len = torch.tensor(question_len, dtype=torch.long)
        prefix_a_len = torch.tensor(prefix_a_len, dtype=torch.long)
        answer_len = torch.tensor(answer_len, dtype=torch.long)
        images = torch.stack(images, dim=0)
        yn_or_oe = torch.tensor(yn_or_oe, dtype=torch.long)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "question_start": question_start,
            "question_len": question_len,
            "prefix_a_len": prefix_a_len,
            "answer_len": answer_len,
            "questions": questions,
            "answers": answers,
            "image_names": image_names,
            "answer_types": answer_types,
            "yn_or_oe": yn_or_oe
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
