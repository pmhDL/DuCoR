import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DEEPSPEED_LOG_LEVEL"] = "warning"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, LlamaTokenizer, GPT2Tokenizer, AutoTokenizer
from trainer.train import Trainer
from trainer.train0 import Trainer0
from trainer.train_pretrain import PretrainTrainer
from trainer.train_sft import SFTTrainer
from test_bs import Tester_BS
from test_sg import Tester_SG
from models import VQAmodel
from data_loaders.dataloader import VQAdataset, VQACollator
from data_loaders.dataloader_test import VQAdataset_test, VQACollator_test
from data_loaders.dataloader_st2 import VQAdataset_st2, VQACollator_st2
from data_loaders.dataloader_pretrain import PretrainDataset, PretrainCollator
from data_loaders.dataloader_sft import InstructionSFTDataset, InstructionSFTCollator
from utils import set_random_seeds
from ducor_config import add_ducor_arguments, apply_method_config, apply_dataset_config



def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='SLAKE', choices=('Path_VQA', 'VQA_RAD', 'SLAKE', 'PMC15M', 'LLaVA_Med_Instruct'))
    parser.add_argument("--method", type=str, default="baseline", choices=("pretrain", "sft", "baseline", "ducor"))
    parser.add_argument("--model_type", type=str, default="stablelm", choices=("llama-2-7b", "mistral-7b", "stablelm", "gpt2-xl"))
    parser.add_argument("--llmlora", type=str, default="lora", choices=("all", "lora", "frozen"))
    parser.add_argument("--vislora", type=str, default="lora", choices=("all", "lora", "frozen"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iters_to_accumulate", type=int, default=1)
    parser.add_argument("--select_layer", type=int, default=-1)
    parser.add_argument("--pad", type=str, default="left", choices=('right', 'left'))
    parser.add_argument("--gen", type=str, default="bs", choices=('bs', 'sig'))
    parser.add_argument("--testsplit", type=str, default="test", choices=('test', 'train'))
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--img_res", type=int, default=384)
    parser.add_argument("--num_beam", type=int, default=1)
    parser.add_argument("--multi_gpu", type=str, default="yes")
    parser.add_argument("--vis_hidden", type=str, default='yes')
    parser.add_argument("--out_dir", default="checkpoints/baseline")
    parser.add_argument("--dataset_path", type=str, default="data/VQA_datasets")
    parser.add_argument("--testst1", type=str, default="experiments/baseline")
    parser.add_argument("--pretrain_path", type=str, default="pre_checkpoints")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Optional checkpoint used to initialize training.")
    parser.add_argument("--pretrain_data_path", type=str, default="data/pmc/llava_med_alignment_500k.json")
    parser.add_argument("--pretrain_image_root", type=str, default="data/pmc/images")
    parser.add_argument("--pretrain_save_every", type=int, default=0)
    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--pretrain_batch_size", type=int, default=16)
    parser.add_argument("--pretrain_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--pretrain_lr", type=float, default=2e-3)
    parser.add_argument("--pretrain_weight_decay", type=float, default=0.0)
    parser.add_argument("--pretrain_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--pretrain_dataloader_num_workers", type=int, default=4)
    parser.add_argument("--sft_data_path", type=str, default="data/pmc/llava_med_instruct_60k_inline_mention_filter.json")
    parser.add_argument("--sft_image_root", type=str, default="data/pmc/images")
    parser.add_argument("--sft_include_history", type=str, default="yes", choices=("yes", "no"))
    parser.add_argument("--sft_max_history_turns", type=int, default=3)
    parser.add_argument("--sft_max_question_tokens", type=int, default=512)
    parser.add_argument("--sft_max_answer_tokens", type=int, default=256)
    parser.add_argument("--sft_save_every", type=int, default=0)
    parser.add_argument("--sft_epochs", type=int, default=5)
    parser.add_argument("--sft_batch_size", type=int, default=16)
    parser.add_argument("--sft_gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--sft_lr", type=float, default=2e-5)
    parser.add_argument("--sft_weight_decay", type=float, default=0.0)
    parser.add_argument("--sft_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--sft_lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--sft_dataloader_num_workers", type=int, default=4)
    parser.add_argument("--vis_path", type=str, default="clip-vit-base-patch16", choices=('clip-vit-base-patch16', 'clip-vit-large-patch14'))
    add_ducor_arguments(parser)
    args = parser.parse_args()
    args = apply_method_config(args)
    args = apply_dataset_config(args)
    set_random_seeds(args.seed)
    return args



if __name__ == "__main__":
    args = parse_argument()
    args.IGNORE_INDEX = -100
    args.IMAGE_TOKEN_INDEX = -200

    suffix = f"dataset_{args.dataset}_llm_{args.model_type}_llmlora_{args.llmlora}_vislora_{args.vislora}_method_{args.method}_pad_{args.pad}"

    args.out_dir = os.path.join(args.out_dir, suffix)
    args.text_path = os.path.join(args.pretrain_path, args.model_type)
    args.vis_path = os.path.join(args.pretrain_path, args.vis_path)
    print(args)

    '''================================ tokenizer =========================================='''
    if 'gpt2' in args.model_type:
        args.tokenizer = GPT2Tokenizer.from_pretrained(args.text_path)
        args.tokenizer.pad_token = args.tokenizer.eos_token
    elif 'llama' in args.model_type:
        args.tokenizer = LlamaTokenizer.from_pretrained(args.text_path)
        args.tokenizer.bos_token = '<|endoftext|>'
        args.tokenizer.eos_token = '<|endoftext|>'
        args.tokenizer.pad_token = '<|endoftext|>'
    elif 'mistral' in args.model_type:
        args.tokenizer = AutoTokenizer.from_pretrained(args.text_path)
        args.tokenizer.bos_token = '<|endoftext|>'
        args.tokenizer.eos_token = '<|endoftext|>'
        args.tokenizer.pad_token = '<|endoftext|>'
    elif 'stablelm' in args.model_type:
        args.tokenizer = AutoTokenizer.from_pretrained(args.text_path, padding_side="right", use_fast=False, )
        args.tokenizer.unk_token = '<|reg0|>'

    args.image_processor = CLIPImageProcessor.from_pretrained(args.vis_path)

    '''================================================dataset========================================================='''
    if args.method == 'pretrain':
        train_dataset = PretrainDataset(args)
        data_collator_pretrain = PretrainCollator(tokenizer=args.tokenizer)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.pretrain_batch_size, shuffle=True, drop_last=False, collate_fn=data_collator_pretrain, num_workers=args.pretrain_dataloader_num_workers, pin_memory=True, prefetch_factor=2,)
        print('len train_dataloader: ', len(train_dataloader))
        print('train samples: ', len(train_dataset))
    elif args.method == 'sft':
        train_dataset = InstructionSFTDataset(args)
        data_collator_sft = InstructionSFTCollator(tokenizer=args.tokenizer)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.sft_batch_size, shuffle=True, drop_last=False, collate_fn=data_collator_sft, num_workers=args.sft_dataloader_num_workers, pin_memory=True, prefetch_factor=2,)
        print('len train_dataloader: ', len(train_dataloader))
        print('train samples: ', len(train_dataset))
    else:
        train_datapath = os.path.join(args.dataset_path, args.dataset)
        test_datapath = os.path.join(args.dataset_path, args.dataset)
        args.testst1 = os.path.join(args.testst1, args.dataset, args.model_type)
        if args.method == 'baseline':
            train_dataset = VQAdataset(args, train_datapath, split='train')
            if args.testsplit == 'test':
                test_dataset = VQAdataset_test(args, test_datapath, split='test')
            elif args.testsplit == 'train':
                test_dataset = VQAdataset_test(args, test_datapath, split='train')

            if args.dataset == 'VQA_RAD':
                val_dataset = VQAdataset(args, train_datapath, split="train")
            else:
                val_dataset = VQAdataset(args, train_datapath, split="val")

            data_collator = VQACollator(tokenizer=args.tokenizer)
            data_collator_test = VQACollator_test(tokenizer=args.tokenizer)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                          drop_last=False, collate_fn=data_collator, num_workers=8, pin_memory=True,
                                          prefetch_factor=2)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                        collate_fn=data_collator, num_workers=8, pin_memory=True, prefetch_factor=2)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                         drop_last=False, collate_fn=data_collator_test)
        elif args.method == 'ducor':
            label_splits = ['train']
            if args.ducor_include_val_in_train == "yes":
                label_splits.append('val')
            train_split = label_splits if len(label_splits) > 1 else label_splits[0]
            train_dataset = VQAdataset_st2(args, train_datapath, args.testst1, split=train_split, psplit='test')
            if args.testsplit == 'test':
                test_dataset = VQAdataset_test(args, test_datapath, split='test')
            elif args.testsplit == 'train':
                test_dataset = VQAdataset_test(args, test_datapath, split='train')

            val_gen_dataset = None
            if args.ducor_include_val_in_train == "yes":
                val_dataset = None
            else:
                if args.dataset == 'VQA_RAD':
                    val_dataset = VQAdataset(args, train_datapath, split="train")
                    val_gen_dataset = VQAdataset_test(args, train_datapath, split="train")
                else:
                    val_dataset = VQAdataset(args, train_datapath, split="val")
                    val_gen_dataset = VQAdataset_test(args, train_datapath, split="val")

            data_collator_st2 = VQACollator_st2(tokenizer=args.tokenizer)
            data_collator_val = VQACollator(tokenizer=args.tokenizer)
            data_collator_test = VQACollator_test(tokenizer=args.tokenizer)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                          drop_last=False, collate_fn=data_collator_st2, num_workers=8, pin_memory=True,
                                          prefetch_factor=2)
            if val_dataset is None:
                val_dataloader = None
                val_gen_dataloader = None
            else:
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                            collate_fn=data_collator_val, num_workers=8, pin_memory=True, prefetch_factor=2)
                val_gen_dataloader = DataLoader(dataset=val_gen_dataset, batch_size=args.batch_size, shuffle=False,
                                                drop_last=False, collate_fn=data_collator_test)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                         drop_last=False, collate_fn=data_collator_test)

        print('len train_dataloader: ', len(train_dataloader))
        print('len val_dataloader: ', 0 if val_dataloader is None else len(val_dataloader))
        print('len test_dataloader: ', len(test_dataloader))
        args.len_train = len(train_dataset)
        args.len_val = 0 if val_dataset is None else len(val_dataset)
        print('train samples: ', len(train_dataset))
        print('val samples: ', 0 if val_dataset is None else len(val_dataset))
        print('test samples: ', len(test_dataset))


    '''=================================================== model ========================================================'''
    model = VQAmodel(args=args)
    if args.phase == 'train':
        if args.init_checkpoint and args.method in ('sft', 'baseline'):
            # checkpoint_pretrain.pt for sft, checkpoint_sft.pt for baseline
            state_dict = torch.load(args.init_checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        if args.method == 'pretrain': # Pretrain starts from base LLM/vision weights
            model = PretrainTrainer(model, train_dataloader, args)
        elif args.method == 'sft': # SFT resumes from checkpoint_pretrain.pt
            model = SFTTrainer(model, train_dataloader, args)
        elif args.method == 'baseline': # Baseline resumes from checkpoint_sft.pt
            model = Trainer0(model, train_dataloader, val_dataloader, args)
        elif args.method == 'ducor':
            pathx = args.out_dir
            pathx = pathx.replace('method_ducor', 'method_baseline')
            checkpoint_path = os.path.join(pathx, "checkpoint_baseline.pt")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model = Trainer(model, train_dataloader, train_dataset, val_dataloader, test_dataloader, args, val_gen_dataloader)
    elif args.phase == 'test':
        checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{args.method}.pt")
        if args.gen == 'bs':
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
            Tester_BS(model, test_dataloader, args)
        elif args.gen == 'sig':
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(args.device)), strict=False)
            Tester_SG(model, test_dataset, args)