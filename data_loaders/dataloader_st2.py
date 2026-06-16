import re
import os
import json
import threading
from dataclasses import dataclass
from typing import Sequence, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
class VQAdataset_st2(Dataset):
    def __init__(self, args, path, ppath, split='train', psplit='test'):
        super(VQAdataset_st2, self).__init__()
        self.IGNORE_INDEX = args.IGNORE_INDEX
        self.IMAGE_TOKEN_INDEX = args.IMAGE_TOKEN_INDEX
        self.args = args
        self.path = path
        self.ppath = ppath
        self.split = split
        if isinstance(split, str):
            self.label_splits = [split]
        else:
            self.label_splits = list(split)
        self.psplit = psplit

        self.lock = threading.Lock()

        imgpath = os.path.join(path, 'images')
        self.jslists_train = []
        for label_split in self.label_splits:
            jspath = os.path.join(path, 'json', label_split + '.json')
            split_items = json.load(open(jspath, 'r'))
            for item in split_items:
                item = dict(item)
                item['image_name'] = os.path.join(imgpath, item['image_name'] + '.jpg')
                item['source_split'] = label_split
                self.jslists_train.append(item)

        self.load_test_data()
        self.tokenizer = args.tokenizer
        self.image_processor = args.image_processor
        self.prefix_q = self.tokenizer('question:', return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_ctx = self.tokenizer(' context:', return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_a = self.tokenizer(' answer:', return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_c = self.tokenizer(' candidate:', return_tensors="pt", add_special_tokens=False).input_ids[0]


    def load_test_data(self):
        jspath_test = os.path.join(self.ppath, self.psplit + '_result.json')
        jslists_test = json.load(open(jspath_test, 'r'))
        Ltrain = len(self.jslists_train)
        Ltest = len(jslists_test)
        with self.lock:
            self.jslists = self.jslists_train + jslists_test
            self.train_test_lb = [0] * Ltrain + [1] * Ltest


    def __len__(self):
        return len(self.jslists)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        itm = self.jslists[index]
        train_or_test = self.train_test_lb[index]
        image_path = itm['image_name']
        image_input = Image.open(image_path).convert('RGB')
        image = self.image_processor(image_input, return_tensors="pt").pixel_values[0]

        question = pre_question(itm['question'])

        if train_or_test == 0:
            ans = pre_answer(itm['answer'])
        else:
            ans = pre_answer(itm['predicted'])
        gtans = pre_answer(itm['answer'])

        if ans == 'yes' or ans == 'no':
            yn_or_oe = 1
        else:
            yn_or_oe = 0

        answer = ans+'<|endoftext|>'
        answertxt = ans

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
        prefix_a_len = len(self.prefix_a)
        answer_len = len(answer_id)
        prefix_c_len = len(self.prefix_c)
        data_dict = dict(input_id=input_id, image=image, question_start=question_start, question_len=question_len,
                         prefix_a_len=prefix_a_len, answer_len=answer_len, prefix_c_len=prefix_c_len, question=question, answer=answertxt,
                         gtans=gtans, image_name=image_path, train_or_test=train_or_test, global_idx=index, yn_or_oe=yn_or_oe)

        return data_dict


@dataclass
class VQACollator_st2(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_id"] for instance in instances]
        images = [instance["image"] for instance in instances]
        question_start = [instance["question_start"] for instance in instances]
        question_len = [instance["question_len"] for instance in instances]
        prefix_a_len = [instance["prefix_a_len"] for instance in instances]
        answer_len = [instance["answer_len"] for instance in instances]
        prefix_c_len = [instance["prefix_c_len"] for instance in instances]
        questions = [instance["question"] for instance in instances]
        answers = [instance["answer"] for instance in instances]
        gtans = [instance["gtans"] for instance in instances]
        image_names = [instance["image_name"] for instance in instances]
        train_or_test = [instance["train_or_test"] for instance in instances]
        global_idx = [instance["global_idx"] for instance in instances]
        yn_or_oe = [instance["yn_or_oe"] for instance in instances]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                        batch_first=True,
                                        padding_value=IGNORE_INDEX)
        padded_input_ids = padded_input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = padded_input_ids.ne(IGNORE_INDEX)


        question_start = torch.tensor(question_start, dtype=torch.long)
        question_len = torch.tensor(question_len, dtype=torch.long)
        images = torch.stack(images, dim=0)  # [B, C, H, W]
        prefix_a_len = torch.tensor(prefix_a_len, dtype=torch.long)
        answer_len = torch.tensor(answer_len, dtype=torch.long)
        train_or_test = torch.tensor(train_or_test, dtype=torch.long)
        global_idx = torch.tensor(global_idx, dtype=torch.long)
        yn_or_oe = torch.tensor(yn_or_oe, dtype=torch.long)


        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "question_start": question_start,
            "question_len": question_len,
            "prefix_a_len": prefix_a_len,
            "answer_len": answer_len,
            "prefix_c_len": prefix_c_len,
            "questions": questions,
            "answers": answers,
            "gtans": gtans,
            "image_names": image_names,
            "train_or_test": train_or_test,
            "global_idx": global_idx,
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
