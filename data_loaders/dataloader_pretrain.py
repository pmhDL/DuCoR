import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def clean_text(text):
    text = str(text or "").replace("<image>", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def conversation_pair(item):
    conversations = item.get("conversations")
    for turn in conversations:
        role = turn.get("from") # human gpt
        value = turn.get("value")
        if role == "human":
            question = value
        elif role == "gpt":
            answer = value
    return question, answer


class PretrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.IGNORE_INDEX = args.IGNORE_INDEX
        self.IMAGE_TOKEN_INDEX = args.IMAGE_TOKEN_INDEX
        self.args = args
        self.image_root = args.pretrain_image_root

        with open(args.pretrain_data_path, "r") as f:
            self.records = json.load(f)

        self.tokenizer = args.tokenizer
        self.image_processor = args.image_processor

        self.prefix_q = self.tokenizer("question:", return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_ctx = self.tokenizer(" context:", return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_a = self.tokenizer(" answer:", return_tensors="pt", add_special_tokens=False).input_ids[0]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        item = self.records[index]
        question, answer_text = conversation_pair(item)
        question = clean_text(question).lower()
        answer_text = answer_text.strip().lower()

        image_path = os.path.join(self.image_root, item['image'])
        image_input = Image.open(image_path).convert("RGB")
        image = self.image_processor(image_input, return_tensors="pt").pixel_values[0]

        answer_for_loss = answer_text + "<|endoftext|>"
        question_id = self.tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids[0]
        answer_id = self.tokenizer(answer_for_loss, return_tensors="pt", add_special_tokens=False).input_ids[0]
        input_id = torch.cat([
            self.prefix_q,
            question_id,
            self.prefix_ctx,
            torch.tensor([self.IMAGE_TOKEN_INDEX], dtype=torch.long),
            self.prefix_a,
            answer_id,
        ], dim=0)

        return {
            "input_id": input_id,
            "image": image,
            "question_start": len(self.prefix_q),
            "question_len": len(question_id),
            "prefix_a_len": len(self.prefix_a),
            "answer_len": len(answer_id),
            "question": question,
            "answer": answer_text,
            "image_name": image_path
        }


@dataclass
class PretrainCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_id"] for instance in instances]
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=IGNORE_INDEX)
        padded_input_ids = padded_input_ids[:, :self.tokenizer.model_max_length]
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_input_ids.ne(IGNORE_INDEX),
            "images": torch.stack([instance["image"] for instance in instances], dim=0),
            "question_start": torch.tensor([instance["question_start"] for instance in instances], dtype=torch.long),
            "question_len": torch.tensor([instance["question_len"] for instance in instances], dtype=torch.long),
            "prefix_a_len": torch.tensor([instance["prefix_a_len"] for instance in instances], dtype=torch.long),
            "answer_len": torch.tensor([instance["answer_len"] for instance in instances], dtype=torch.long),
            "questions": [instance["question"] for instance in instances],
            "answers": [instance["answer"] for instance in instances],
            "image_names": [instance["image_name"] for instance in instances]
        }
