import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaForCausalLM, StableLmForCausalLM, GPT2LMHeadModel, CLIPVisionModel
from peft import LoraConfig, get_peft_model, TaskType
from mappers import Linear_proj, QueryAggregator, ProjectionHead
from vit import VisionTransformer
from utils import load_checkpoint, clean_answer


class VQAmodel(nn.Module):
    def __init__(self, args):
        super(VQAmodel, self).__init__()
        #------------------------------parameters---------------------------
        self.args = args
        self.model_type = args.model_type

        # ------------------------tokenizer and text encoder------------------
        if 'gpt2' in self.model_type:
            self.text_encoder = GPT2LMHeadModel.from_pretrained(args.text_path)
            self.text_encoder.config.pad_token_id = self.args.tokenizer.pad_token_id
            self.llm_emb_size = self.text_encoder.config.n_embd
        elif 'llama' in self.model_type:
            self.text_encoder = LlamaForCausalLM.from_pretrained(args.text_path)
            self.llm_emb_size = self.text_encoder.config.hidden_size
        elif 'stablelm' in self.model_type:
            self.text_encoder = StableLmForCausalLM.from_pretrained(args.text_path)
            self.llm_emb_size = self.text_encoder.config.hidden_size
        elif 'mistral' in args.model_type:
            self.text_encoder = AutoModelForCausalLM.from_pretrained(args.text_path)
            self.llm_emb_size = self.text_encoder.config.hidden_size

        # load the fine-tuning strategy
        if self.args.llmlora == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1, bias="none")
            self.text_encoder = get_peft_model(self.text_encoder, peft_config)
            self.text_encoder.print_trainable_parameters()
        elif self.args.llmlora == 'frozen':
            for param in self.text_encoder.transformer.parameters():
                param.requires_grad = False

        #-----------------------------visual encoder---------------------------
        if 'clip' in self.args.vis_path:
            self.image_encoder = CLIPVisionModel.from_pretrained(args.vis_path)
            target_modules = ["q_proj", "v_proj"]
            self.vis_emb_size = self.image_encoder.config.hidden_size
        elif 'vision' in self.args.vis_path:
            target_modules = ["qkv"]
            vitmodel = VisionTransformer(img_size=self.args.img_res)
            self.image_encoder, msg = load_checkpoint(vitmodel, self.args.vis_path)
            self.vis_emb_size = 768

        if self.args.vislora=='lora':
            loraconfig = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules = target_modules, bias="none",)
            self.image_encoder = get_peft_model(self.image_encoder, loraconfig)
            self.image_encoder.print_trainable_parameters()
        elif self.args.vislora=='frozen':
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        #---------------------- query aggregate----------------------------
        if self.args.seq_sim == 'query':
            self.query_aggregator = QueryAggregator(self.llm_emb_size)
        elif self.args.seq_sim == 'mean':
            self.mean_aggregator = ProjectionHead(self.llm_emb_size)

        #------------------------------------projector----------------------------------
        self.img_project = Linear_proj(self.vis_emb_size, self.llm_emb_size)


    def token_embedder(self):
        if 'gpt2' in self.model_type:
            return self.text_encoder.transformer.wte
        if 'stablelm' in self.model_type or 'llama' in self.model_type or 'mistral' in self.model_type:
            if self.args.llmlora == 'lora':
                return self.text_encoder.base_model.model.model.embed_tokens
            return self.text_encoder.model.embed_tokens


    def image_embeddings(self, batch, device, key='images'):
        image = batch[key]
        image = image.to(device=device, dtype=next(self.image_encoder.parameters()).dtype)
        if key == 'image':
            image = image.unsqueeze(0)
        if self.args.vis_hidden == 'yes' and 'vision' not in self.args.vis_path:
            img_feat = self.image_encoder(image, output_hidden_states=True)
        else:
            img_feat = self.image_encoder(image)
        if 'vision' in self.args.vis_path:
            return img_feat
        if self.args.vis_hidden == 'yes':
            return img_feat.hidden_states[self.args.select_layer]
        return img_feat.last_hidden_state


    def position_ids(self, masks):
        position_ids = masks.to(torch.long).cumsum(-1) - 1
        return position_ids.clamp_min(0)


    def answer_ce_losses(self, logits, sequences, ans_starts, len_ans):
        loss = []
        for b in range(logits.size(0)):
            condensed_tokens = sequences[b, ans_starts[b]:ans_starts[b] + len_ans[b]]
            condensed_logits = logits[b, ans_starts[b] - 1: ans_starts[b] + len_ans[b] - 1]
            loss.append(F.cross_entropy(condensed_logits.view(-1, logits.shape[-1]), condensed_tokens.view(-1)))
        return loss


    def forward(self, batch, device):
        token_embder = self.token_embedder()
        imgemb = self.image_embeddings(batch, device)

        if self.args.pad == 'right':
            embbedings, masks, sequences, ans_starts, len_ans, _, _ = self.prepare_multimodal_inputs_train(batch, imgemb, device, token_embder)
            outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks)
        elif self.args.pad == 'left':
            embbedings, masks, sequences, ans_starts, len_ans, _, _ = self.prepare_multimodal_inputs_train_left(batch, imgemb, device, token_embder)
            position_ids = self.position_ids(masks)
            outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, position_ids=position_ids)

        return self.answer_ce_losses(outputs.logits, sequences, ans_starts, len_ans)


    def forward_sft(self, batch, device):
        token_embder = self.token_embedder()
        imgemb = self.image_embeddings(batch, device)

        if self.args.pad == 'right':
            embbedings, masks, labels = self.prepare_multimodal_inputs_sft(batch, imgemb, device, token_embder)
            outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, labels=labels)
        elif self.args.pad == 'left':
            embbedings, masks, labels = self.prepare_multimodal_inputs_sft_left(batch, imgemb, device, token_embder)
            position_ids = self.position_ids(masks)
            outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, position_ids=position_ids, labels=labels)
        return outputs.loss


    def forward_st2(self, batch, device, protodict=None):
        token_embder = self.token_embedder()
        imgemb = self.image_embeddings(batch, device)

        if self.args.seq_sim == 'query':
            aggregator = self.query_aggregator
        elif self.args.seq_sim == 'mean':
            aggregator = self.mean_aggregator

        if self.args.method == 'ducor':
            if self.args.pad == 'right':
                embbedings, masks, sequences, ans_starts, len_ans, end_iq, end_ans = self.prepare_multimodal_inputs_train(batch, imgemb, device, token_embder)
                outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, output_hidden_states=True)
                last_feat = outputs.hidden_states[-1]
                if not protodict:
                    aggembs = self.com_aggemb(last_feat, aggregator, end_iq, self.args.seq_sim)
                    loss_ctrs = [emb.sum() * 0.0 for emb in aggembs]
                    self.cr_skip_stats = {"missing": 0, "total": 0, "examples": [], "disabled": True}
                else:
                    loss_ctrs, aggembs = contrastive_aggemb(protodict, last_feat, batch, aggregator, end_iq, sim=self.args.seq_sim, temperature=self.args.temperature)
                    self.cr_skip_stats = getattr(contrastive_aggemb, "last_stats", {})
            elif self.args.pad == 'left':
                embbedings, masks, sequences, ans_starts, len_ans, start_iq, end_iq = self.prepare_multimodal_inputs_train_left(batch, imgemb, device, token_embder)
                position_ids = self.position_ids(masks)
                outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, position_ids=position_ids, output_hidden_states=True)
                last_feat = outputs.hidden_states[-1]
                if not protodict:
                    aggembs = self.com_aggemb_left(last_feat, aggregator, start_iq, end_iq, self.args.seq_sim)
                    loss_ctrs = [emb.sum() * 0.0 for emb in aggembs]
                    self.cr_skip_stats = {"missing": 0, "total": 0, "examples": [], "disabled": True}
                else:
                    loss_ctrs, aggembs = contrastive_aggemb_left(protodict, last_feat, batch, aggregator, start_iq, end_iq, sim=self.args.seq_sim, temperature=self.args.temperature)
                    self.cr_skip_stats = getattr(contrastive_aggemb_left, "last_stats", {})
            loss = self.answer_ce_losses(outputs.logits, sequences, ans_starts, len_ans)
            return loss, loss_ctrs, aggembs

        elif self.args.method == 'baseline':
            if self.args.pad == 'right':
                embbedings, masks, sequences, ans_starts, len_ans, end_iq, _ = self.prepare_multimodal_inputs_train(batch, imgemb, device, token_embder)
                outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, output_hidden_states=True)
                last_feat = outputs.hidden_states[-1]
                qi_emb = self.com_aggemb(last_feat, aggregator, end_iq, self.args.seq_sim)
            elif self.args.pad == 'left':
                embbedings, masks, sequences, ans_starts, len_ans, start_iq, end_iq = self.prepare_multimodal_inputs_train_left(batch, imgemb, device, token_embder)
                position_ids = self.position_ids(masks)
                outputs = self.text_encoder(inputs_embeds=embbedings, attention_mask=masks, position_ids=position_ids, output_hidden_states=True)
                last_feat = outputs.hidden_states[-1]
                qi_emb = self.com_aggemb_left(last_feat, aggregator, start_iq, end_iq, self.args.seq_sim)
            loss = self.answer_ce_losses(outputs.logits, sequences, ans_starts, len_ans)

            return loss, qi_emb


    def generate_bs(self, batch, device):
        token_embder = self.token_embedder()
        imgemb = self.image_embeddings(batch, device)

        if self.args.pad == 'right':
            batch_pred = []
            for idd in range(len(imgemb)):
                input_emb, masks, sequences = self.prepare_multimodal_inputs_test_single(idd, batch, imgemb[idd].unsqueeze(0), device, token_embder)
                output_ids = self.text_encoder.generate(inputs_embeds=input_emb, attention_mask=masks,
                                                        max_new_tokens=20,
                                                        pad_token_id=self.args.tokenizer.pad_token_id)
                predicted_ans = self.args.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                if 'llama' in self.model_type or 'mistral' in self.model_type:
                    predicted_ans = clean_answer(predicted_ans).split('<')[0].replace("� ", "")
                batch_pred.append(predicted_ans)
            return batch_pred
        elif self.args.pad == 'left':
            input_emb, masks, sequences = self.prepare_multimodal_inputs_test_bs_left(batch, imgemb, device, token_embder)
            position_ids = self.position_ids(masks)
            output_ids = self.text_encoder.generate(inputs_embeds=input_emb, attention_mask=masks, position_ids=position_ids,
                                                    max_new_tokens=20, pad_token_id=self.args.tokenizer.pad_token_id,
                                                    eos_token_id=self.args.tokenizer.eos_token_id)
            batch_predicted = self.args.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if 'llama' in self.model_type or 'mistral' in self.model_type:
                batch_predicted = [clean_answer(pred.strip()).split('<')[0] for pred in batch_predicted]
            return batch_predicted


    def generate(self, batch, device):
        token_embder = self.token_embedder()
        imgemb = self.image_embeddings(batch, device, key='image')
        input_emb, masks, sequences = self.prepare_multimodal_inputs_test(batch, imgemb, device, token_embder)

        if self.args.num_beam > 1:
            output_ids = self.text_encoder.generate(inputs_embeds=input_emb, attention_mask=masks,
                                                    max_new_tokens=20,
                                                    pad_token_id=self.args.tokenizer.pad_token_id,
                                                    num_beams=self.args.num_beam,
                                                    num_return_sequences=self.args.num_beam)
            beam_pred_ans = self.args.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            beam_pred = [clean_answer(beam_pred_ans[k].strip()) for k in range(self.args.num_beam)]
            if 'llama' in self.model_type or 'mistral' in self.model_type:
                beam_pred = [beam_predi.split('<')[0].replace("� ", "") for beam_predi in beam_pred]
            return beam_pred
        else:
            output_ids = self.text_encoder.generate(inputs_embeds=input_emb, attention_mask=masks,
                                                    max_new_tokens=20,
                                                    pad_token_id=self.args.tokenizer.pad_token_id)
            predicted_ans = self.args.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if 'llama' in self.model_type or 'mistral' in self.model_type:
                predicted_ans = clean_answer(predicted_ans).split('<')[0].replace("� ", "")
            return predicted_ans



    def prepare_multimodal_inputs_train(self, batch, imgemb, device, token_embder):
        bs = len(imgemb)
        imgemb = self.img_project(imgemb)

        Input_Emb = []
        Masks = []
        Seq = []
        Ans_starts = []
        Len_ans = []
        end_iq = []
        end_ans = []
        prefix_a_len = batch['prefix_a_len'][0]

        for ii in range(bs):
            answer_len = batch['answer_len'][ii]
            Len_ans.append(answer_len)
            input_ids = batch['input_ids'][ii]
            mask0 = input_ids.ne(self.args.IGNORE_INDEX)
            input_ids = input_ids[mask0]

            img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
            input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id+1:])).long()
            embedding = token_embder(input_ids_temp)

            image_emb = imgemb[ii]
            sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long),input_ids[img_place_id+1:])).type(torch.long)
            Seq.append(sequence_all)
            ans_start_id = img_place_id + image_emb.size(0) + prefix_a_len
            Ans_starts.append(ans_start_id)
            end_iq.append(img_place_id + image_emb.size(0))
            end_ans.append(sequence_all.size(0))

            input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
            Input_Emb.append(input_embedding)
            mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
            Masks.append(mask)

        end_iq = torch.tensor(end_iq)
        end_ans = torch.tensor(end_ans)
        max_len = max(x.size(0) for x in Input_Emb)
        emb_padded = []
        mask_padded = []
        seq_padded = []

        for kk in range(bs):
            cur_embed = Input_Emb[kk]
            cur_len = cur_embed.size(0)
            pademb = torch.cat((cur_embed, torch.zeros((max_len - cur_len, cur_embed.size(1)), dtype=cur_embed.dtype, device=device)), dim=0)
            emb_padded.append(pademb)

            cur_mask = Masks[kk]
            padmask = torch.cat((cur_mask, torch.zeros((max_len - cur_len), dtype=cur_mask.dtype, device=device)))
            mask_padded.append(padmask)

            cur_seq = Seq[kk]
            padseq = torch.cat((cur_seq, torch.zeros((max_len - cur_len), dtype=cur_seq.dtype, device=device)))
            seq_padded.append(padseq)

        emb_padded = torch.stack(emb_padded, dim=0)
        mask_padded = torch.stack(mask_padded, dim=0)
        seq_padded = torch.stack(seq_padded, dim=0)
        Ans_starts = torch.tensor(Ans_starts, device=device)
        Len_ans = torch.tensor(Len_ans, device=device)
        return emb_padded, mask_padded, seq_padded, Ans_starts, Len_ans, end_iq, end_ans


    def prepare_multimodal_inputs_train_left(self, batch, imgemb, device, token_embder):
        bs = len(imgemb)
        imgemb = self.img_project(imgemb)

        Input_Emb = []
        Masks = []
        Seq = []
        Ans_starts = []
        Len_ans = []
        end_iq = []
        prefix_a_len = batch['prefix_a_len'][0]
        for ii in range(bs):
            answer_len = batch['answer_len'][ii]
            Len_ans.append(answer_len)
            input_ids = batch['input_ids'][ii]
            mask0 = input_ids.ne(self.args.IGNORE_INDEX)
            input_ids = input_ids[mask0]

            img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
            input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id+1:])).long()
            embedding = token_embder(input_ids_temp)

            image_emb = imgemb[ii]
            sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long),input_ids[img_place_id+1:])).type(torch.long)
            Seq.append(sequence_all)
            ans_start_id = img_place_id + image_emb.size(0) + prefix_a_len
            Ans_starts.append(ans_start_id)
            end_iq.append(img_place_id + image_emb.size(0))

            input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
            Input_Emb.append(input_embedding)
            mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
            Masks.append(mask)

        max_len = max(x.size(0) for x in Input_Emb)
        emb_padded = []
        mask_padded = []
        seq_padded = []
        Ans_starts1 = []
        end_iq1 = []
        start_iq = []
        for kk in range(bs):
            cur_embed = Input_Emb[kk]
            cur_len = cur_embed.size(0)
            pad_len = max_len - cur_len
            start_iq.append(pad_len)
            pademb = torch.cat((torch.zeros((pad_len, cur_embed.size(1)), dtype=cur_embed.dtype, device=device), cur_embed), dim=0)
            emb_padded.append(pademb)

            cur_mask = Masks[kk]
            padmask = torch.cat((torch.zeros((pad_len), dtype=cur_mask.dtype, device=device), cur_mask))
            mask_padded.append(padmask)

            cur_seq = Seq[kk]
            padseq = torch.cat((torch.zeros((pad_len), dtype=cur_seq.dtype, device=device), cur_seq))
            seq_padded.append(padseq)

            Ans_starts1.append(pad_len + Ans_starts[kk])
            end_iq1.append(pad_len + end_iq[kk])
        emb_padded = torch.stack(emb_padded, dim=0)
        mask_padded = torch.stack(mask_padded, dim=0)
        seq_padded = torch.stack(seq_padded, dim=0)
        Ans_starts1 = torch.tensor(Ans_starts1, device=device)
        end_iq1 = torch.tensor(end_iq1, device=device)
        Len_ans = torch.tensor(Len_ans, device=device)
        start_iq = torch.tensor(start_iq, device=device)
        return emb_padded, mask_padded, seq_padded, Ans_starts1, Len_ans, start_iq, end_iq1


    def prepare_multimodal_inputs_sft(self, batch, imgemb, device, token_embder):
        bs = len(imgemb)
        imgemb = self.img_project(imgemb)

        Input_Emb = []
        Masks = []
        Labels = []

        for ii in range(bs):
            input_ids = batch['input_ids'][ii].to(device)
            labels = batch['labels'][ii].to(device)
            mask0 = input_ids.ne(self.args.IGNORE_INDEX)
            input_ids = input_ids[mask0]
            labels = labels[mask0]

            img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
            input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id + 1:])).long()
            embedding = token_embder(input_ids_temp)

            image_emb = imgemb[ii]
            input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
            image_labels = torch.full((image_emb.size(0),), self.args.IGNORE_INDEX, device=device, dtype=torch.long)
            label_all = torch.cat((labels[:img_place_id], image_labels, labels[img_place_id + 1:])).long()

            Input_Emb.append(input_embedding)
            Labels.append(label_all)
            Masks.append(torch.ones(input_embedding.size(0), device=device, dtype=torch.bool))

        max_len = max(x.size(0) for x in Input_Emb)
        emb_padded = []
        mask_padded = []
        label_padded = []

        for kk in range(bs):
            cur_embed = Input_Emb[kk]
            cur_len = cur_embed.size(0)
            pad_len = max_len - cur_len
            pademb = torch.cat((cur_embed, torch.zeros((pad_len, cur_embed.size(1)), dtype=cur_embed.dtype, device=device)), dim=0)
            emb_padded.append(pademb)

            cur_mask = Masks[kk]
            padmask = torch.cat((cur_mask, torch.zeros((pad_len), dtype=cur_mask.dtype, device=device)))
            mask_padded.append(padmask)

            cur_labels = Labels[kk]
            padlabels = torch.cat((cur_labels, torch.full((pad_len,), self.args.IGNORE_INDEX, dtype=cur_labels.dtype, device=device)))
            label_padded.append(padlabels)

        emb_padded = torch.stack(emb_padded, dim=0)
        mask_padded = torch.stack(mask_padded, dim=0)
        label_padded = torch.stack(label_padded, dim=0)
        return emb_padded, mask_padded, label_padded


    def prepare_multimodal_inputs_sft_left(self, batch, imgemb, device, token_embder):
        bs = len(imgemb)
        imgemb = self.img_project(imgemb)

        Input_Emb = []
        Masks = []
        Labels = []

        for ii in range(bs):
            input_ids = batch['input_ids'][ii].to(device)
            labels = batch['labels'][ii].to(device)
            mask0 = input_ids.ne(self.args.IGNORE_INDEX)
            input_ids = input_ids[mask0]
            labels = labels[mask0]

            img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
            input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id + 1:])).long()
            embedding = token_embder(input_ids_temp)

            image_emb = imgemb[ii]
            input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
            image_labels = torch.full((image_emb.size(0),), self.args.IGNORE_INDEX, device=device, dtype=torch.long)
            label_all = torch.cat((labels[:img_place_id], image_labels, labels[img_place_id + 1:])).long()

            Input_Emb.append(input_embedding)
            Labels.append(label_all)
            Masks.append(torch.ones(input_embedding.size(0), device=device, dtype=torch.bool))

        max_len = max(x.size(0) for x in Input_Emb)
        emb_padded = []
        mask_padded = []
        label_padded = []

        for kk in range(bs):
            cur_embed = Input_Emb[kk]
            cur_len = cur_embed.size(0)
            pad_len = max_len - cur_len
            pademb = torch.cat((torch.zeros((pad_len, cur_embed.size(1)), dtype=cur_embed.dtype, device=device), cur_embed), dim=0)
            emb_padded.append(pademb)

            cur_mask = Masks[kk]
            padmask = torch.cat((torch.zeros((pad_len), dtype=cur_mask.dtype, device=device), cur_mask))
            mask_padded.append(padmask)

            cur_labels = Labels[kk]
            padlabels = torch.cat((torch.full((pad_len,), self.args.IGNORE_INDEX, dtype=cur_labels.dtype, device=device), cur_labels))
            label_padded.append(padlabels)

        emb_padded = torch.stack(emb_padded, dim=0)
        mask_padded = torch.stack(mask_padded, dim=0)
        label_padded = torch.stack(label_padded, dim=0)
        return emb_padded, mask_padded, label_padded


    def prepare_multimodal_inputs_test(self, batch, imgemb, device, token_embder):
        imgemb = self.img_project(imgemb)

        input_ids = batch['input_id'].to(device)
        mask0 = input_ids.ne(self.args.IGNORE_INDEX)
        input_ids = input_ids[mask0]

        img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
        input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id + 1:]))
        embedding = token_embder(input_ids_temp)

        image_emb = imgemb[0]
        sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long), input_ids[img_place_id+1:]))
        input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
        mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
        input_embeddings = input_embedding.unsqueeze(0)
        masks = mask.unsqueeze(0)
        sequences = sequence_all.unsqueeze(0)
        return input_embeddings, masks, sequences


    def prepare_multimodal_inputs_test_single(self, id, batch, imgemb, device, token_embder):
        imgemb = self.img_project(imgemb)

        input_ids = batch['input_ids'][id].to(device)
        mask0 = input_ids.ne(self.args.IGNORE_INDEX)
        input_ids = input_ids[mask0]

        img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
        input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id + 1:]))
        embedding = token_embder(input_ids_temp)

        image_emb = imgemb[0]

        sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long), input_ids[img_place_id+1:]))
        input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
        mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool)
        input_embeddings = input_embedding.unsqueeze(0)
        masks = mask.unsqueeze(0)
        sequences = sequence_all.unsqueeze(0)
        return input_embeddings, masks, sequences


    def prepare_multimodal_inputs_test_bs_left(self, batch, imgemb, device, token_embder):
        bs = len(imgemb)
        imgemb = self.img_project(imgemb)

        Input_Emb = []
        Masks = []
        Seq = []
        for ii in range(bs):
            input_ids = batch['input_ids'][ii].to(device)
            mask0 = input_ids.ne(self.args.IGNORE_INDEX)
            input_ids = input_ids[mask0]

            img_place_id = torch.where(input_ids == self.args.IMAGE_TOKEN_INDEX)[0].item()
            input_ids_temp = torch.cat((input_ids[:img_place_id], input_ids[img_place_id+1:])).long()
            embedding = token_embder(input_ids_temp)

            image_emb = imgemb[ii]
            sequence_all = torch.cat((input_ids[:img_place_id], torch.full((image_emb.size(0),), self.args.IMAGE_TOKEN_INDEX, device=device, dtype=torch.long),input_ids[img_place_id+1:])).type(torch.long)
            Seq.append(sequence_all)
            input_embedding = torch.cat((embedding[:img_place_id], image_emb, embedding[img_place_id:]))
            Input_Emb.append(input_embedding)
            mask = torch.ones(input_embedding.size(0), device=device, dtype=torch.bool) 
            Masks.append(mask)

        max_len = max(x.size(0) for x in Input_Emb)
        emb_padded = []
        mask_padded = []
        seq_padded = []

        for kk in range(bs):
            cur_embed = Input_Emb[kk]
            cur_len = cur_embed.size(0)
            pad_len = max_len - cur_len
            pademb = torch.cat((torch.zeros((pad_len, cur_embed.size(1)), dtype=cur_embed.dtype, device=device), cur_embed), dim=0)
            emb_padded.append(pademb)

            cur_mask = Masks[kk]
            padmask = torch.cat((torch.zeros((pad_len), dtype=cur_mask.dtype, device=device), cur_mask))
            mask_padded.append(padmask)

            cur_seq = Seq[kk]
            padseq = torch.cat((torch.zeros((pad_len), dtype=cur_seq.dtype, device=device), cur_seq))
            seq_padded.append(padseq)

        emb_padded = torch.stack(emb_padded, dim=0)
        mask_padded = torch.stack(mask_padded, dim=0)
        seq_padded = torch.stack(seq_padded, dim=0)
        return emb_padded, mask_padded, seq_padded


    def com_aggemb(self, last_feat, aggregator, end_iq, sim='mean'):
        B = last_feat.size(0)
        qi_emb = []
        for i in range(B):
            if sim == 'mean':
                vec = aggregator(last_feat[i, :end_iq[i]].mean(0, keepdim=True))
            elif sim == 'query':
                vec = aggregator(last_feat[i, :end_iq[i]].unsqueeze(0))
            qi_emb.append(vec)
        qi_emb = torch.cat(qi_emb, dim=0)
        qi_emb = F.normalize(qi_emb, dim=-1)
        return qi_emb


    def com_aggemb_left(self, last_feat, aggregator, start_iq, end_iq, sim='mean'):
        B = last_feat.size(0)
        qi_emb = []
        for i in range(B):
            if sim == 'mean':
                vec = aggregator(last_feat[i, start_iq[i]:end_iq[i]].mean(0, keepdim=True))
            elif sim == 'query':
                vec = aggregator(last_feat[i, start_iq[i]:end_iq[i]].unsqueeze(0))
            qi_emb.append(vec)
        qi_emb = torch.cat(qi_emb, dim=0)  # [B, D]
        qi_emb = F.normalize(qi_emb, dim=-1)
        return qi_emb


def contrastive_aggemb(protodict, last_feat, batch, aggregator, end_iq, sim='mean', temperature=0.1):
    device = last_feat.device
    B = last_feat.size(0)
    qi_emb = []
    for i in range(B):
        if sim == 'mean':
            vec = aggregator(last_feat[i, :end_iq[i]].mean(0, keepdim=True))
        elif sim == 'query':
            vec = aggregator(last_feat[i, :end_iq[i]].unsqueeze(0))
        qi_emb.append(vec)
    qi_emb = torch.cat(qi_emb, dim=0)
    qi_emb = F.normalize(qi_emb, dim=-1)

    answers = batch['answers']
    yn_or_oe = batch['yn_or_oe']

    all_keys = list(protodict.keys())
    close_keys = ['yes', 'no']
    open_keys  = [k for k in all_keys if k not in close_keys]
    proto_close = torch.cat([torch.as_tensor(protodict[k]) for k in close_keys], dim=0).to(device)
    proto_open  = torch.cat([torch.as_tensor(protodict[k]) for k in open_keys], dim=0).to(device)

    bs_loss = []
    missing_answers = []
    for i in range(B):
        q = qi_emb[i].unsqueeze(0)
        yi = answers[i]
        if yn_or_oe[i] == 1:
            cand_keys = close_keys
            cand_proto = proto_close
        else:
            cand_keys = open_keys
            cand_proto = proto_open

        if yi not in cand_keys:
            missing_answers.append(str(yi))
            bs_loss.append(q.sum() * 0.0)
            continue

        target_idx = cand_keys.index(yi)
        cand_proto = cand_proto.to(dtype=q.dtype, device=device)
        logits = (q @ cand_proto.t()) / temperature
        target = torch.tensor([target_idx], device=device, dtype=torch.long)
        loss_i = F.cross_entropy(logits, target, reduction='mean')
        bs_loss.append(loss_i)

    contrastive_aggemb.last_stats = {
        "missing": len(missing_answers),
        "total": B,
        "examples": missing_answers[:5],
    }
    return bs_loss, qi_emb


def contrastive_aggemb_left(protodict, last_feat, batch, aggregator, start_iq, end_iq, sim='mean', temperature=0.1):
    device = last_feat.device
    B = last_feat.size(0)
    qi_emb = []
    for i in range(B):
        if sim == 'mean':
            vec = aggregator(last_feat[i, start_iq[i]:end_iq[i]].mean(0, keepdim=True))
        elif sim == 'query':
            vec = aggregator(last_feat[i, start_iq[i]:end_iq[i]].unsqueeze(0))
        qi_emb.append(vec)
    qi_emb = torch.cat(qi_emb, dim=0)
    qi_emb = F.normalize(qi_emb, dim=-1)

    answers = batch['answers']
    yn_or_oe = batch['yn_or_oe']

    all_keys = list(protodict.keys())
    close_keys = ['yes', 'no']
    open_keys  = [k for k in all_keys if k not in close_keys]
    proto_close = torch.cat([torch.as_tensor(protodict[k]) for k in close_keys], dim=0).to(device)
    proto_open  = torch.cat([torch.as_tensor(protodict[k]) for k in open_keys], dim=0).to(device)

    bs_loss = []
    missing_answers = []
    for i in range(B):
        q = qi_emb[i].unsqueeze(0)
        yi = answers[i]
        if yn_or_oe[i] == 1:
            cand_keys = close_keys
            cand_proto = proto_close
        else:
            cand_keys = open_keys
            cand_proto = proto_open

        if yi not in cand_keys:
            missing_answers.append(str(yi))
            bs_loss.append(q.sum() * 0.0)
            continue

        target_idx = cand_keys.index(yi)
        cand_proto = cand_proto.to(dtype=q.dtype, device=device)
        logits = (q @ cand_proto.t()) / temperature
        target = torch.tensor([target_idx], device=device, dtype=torch.long)
        loss_i = F.cross_entropy(logits, target, reduction='mean')
        bs_loss.append(loss_i)

    contrastive_aggemb_left.last_stats = {
        "missing": len(missing_answers),
        "total": B,
        "examples": missing_answers[:5],
    }
    return bs_loss, qi_emb
