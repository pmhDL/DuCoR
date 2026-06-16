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
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

class InstructionSFTDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.IGNORE_INDEX = args.IGNORE_INDEX
        self.IMAGE_TOKEN_INDEX = args.IMAGE_TOKEN_INDEX
        self.args = args
        self.image_root = args.sft_image_root

        with open(args.sft_data_path, "r") as f:
            self.records = json.load(f)

        self.tokenizer = args.tokenizer
        self.image_processor = args.image_processor

        self.prefix_ctx = self.tokenizer("context:", return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_q = self.tokenizer(" question:", return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.prefix_a = self.tokenizer(" answer:", return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.eos_ids = self.tokenizer("<|endoftext|>", return_tensors="pt", add_special_tokens=False).input_ids[0]
        self.max_question_tokens = args.sft_max_question_tokens
        self.max_answer_tokens = args.sft_max_answer_tokens

    def __len__(self):
        return len(self.records)

    def tokenize_question(self, text):
        token_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        token_ids = token_ids[-self.max_question_tokens:]
        return token_ids

    def tokenize_answer(self, text):
        token_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        token_ids = token_ids[:self.max_answer_tokens]
        return torch.cat([token_ids, self.eos_ids], dim=0)

    def build_multiturn_sequence(self, conversations):
        input_parts = []
        label_parts = []
        Answers = []
        image_prefix = torch.cat([self.prefix_ctx, torch.tensor([self.IMAGE_TOKEN_INDEX], dtype=torch.long),], dim=0)
        image_labels = torch.full((len(self.prefix_ctx) + 1,), IGNORE_INDEX, dtype=torch.long)
        input_parts.append(image_prefix)
        label_parts.append(image_labels)

        for turn in conversations:
            role = turn["from"].lower()
            text = clean_text(turn["value"])

            if role == "human":
                question = text
                question_ids = self.tokenize_question(question)

            elif role == "gpt":
                answer = text
                answer_ids = self.tokenize_answer(answer)

                turn_input = torch.cat([
                    self.prefix_q,
                    question_ids,
                    self.prefix_a,
                    answer_ids,
                ], dim=0)

                turn_labels = torch.cat([
                    torch.full(
                        (len(self.prefix_q) + len(question_ids) + len(self.prefix_a),),
                        IGNORE_INDEX,
                        dtype=torch.long,
                    ),
                    answer_ids,
                ], dim=0)

                input_parts.append(turn_input)
                label_parts.append(turn_labels)
                Answers.append(answer)

        return torch.cat(input_parts, dim=0), torch.cat(label_parts, dim=0), Answers

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        item = self.records[index]
        image_path = os.path.join(self.image_root, item["image"])
        image_input = Image.open(image_path).convert("RGB")
        image = self.image_processor(image_input, return_tensors="pt").pixel_values[0]

        conversations = item.get("conversations")
        input_id, labels, answers = self.build_multiturn_sequence(conversations)

        return {
            "input_id": input_id,
            "labels": labels,
            "image": image,
            "answers_all": answers,
            "image_name": image_path
        }


@dataclass
class InstructionSFTCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_id"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=IGNORE_INDEX)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        padded_input_ids = padded_input_ids[:, :self.tokenizer.model_max_length]
        padded_labels = padded_labels[:, :self.tokenizer.model_max_length]
        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": padded_input_ids.ne(IGNORE_INDEX),
            "images": torch.stack([instance["image"] for instance in instances], dim=0),
            "answers_all": [instance["answers_all"] for instance in instances],
            "image_names": [instance["image_name"] for instance in instances],
        }
