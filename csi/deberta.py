import os
import json
import pickle
import random
import argparse
from typing import Tuple, Optional, Union

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
device = torch.device('cuda:0')
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoConfig, AutoTokenizer, DebertaV2ForMaskedLM, DebertaV2Model
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import MaskedLMOutput


def to_device(obj, dev):
    if isinstance(obj, dict):
        return {k: to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, dev) for v in obj]
    if isinstance(obj, tuple):
        return tuple([to_device(v, dev) for v in obj])
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    return obj


@dataclass
class SpeakerLabelingOutput(MaskedLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    indices: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DebertaV3ForSpeakerLabeling(DebertaV2ForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bos_token_id = config.bos_token_id
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(0.5)
        self.sim_head = nn.Sequential(
            nn.Linear(config.hidden_size*3, config.hidden_size),
            nn.GELU(), nn.Linear(config.hidden_size, 1)
        )
        self.post_init()

    @torch.autocast()
    def forward(
        self,
        temperature: Optional[float] = 1.,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        '''
        labels:
            len(labels) = batch_size, where each element is [[i1, j1], [i2, j2], ...]. indicates i1-th and j1-th utterance have the same speaker.
        '''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if labels is None:
            # inference mode
            selected_hidden_state_list, logits_list = list(), list()
            for i, (hidden_state, input_id) in enumerate(zip(last_hidden_state, input_ids)):
                indices = input_id == self.bos_token_id
                selected_hidden_state = hidden_state[indices]
                num_sents, hidden_size = selected_hidden_state.size()

                concatenated_hidden_state = torch.zeros(num_sents, num_sents, hidden_size*3, device=selected_hidden_state.device)
                concatenated_hidden_state[:, :, :hidden_size] = selected_hidden_state.unsqueeze(1)
                concatenated_hidden_state[:, :, hidden_size:hidden_size*2] = selected_hidden_state.unsqueeze(0)
                concatenated_hidden_state[:, :, hidden_size*2:hidden_size*3] = torch.abs(selected_hidden_state.unsqueeze(0) - selected_hidden_state.unsqueeze(1))

                logits = self.sim_head(self.dropout(concatenated_hidden_state)).squeeze()
                selected_hidden_state_list.append(selected_hidden_state)
                logits_list.append(logits)
            loss = None
            return selected_hidden_state_list, logits_list

        else:
            # training mode
            selected_hidden_state_list = list()
            for hidden_state, input_id in zip(last_hidden_state, input_ids):
                indices = input_id == self.bos_token_id
                selected_hidden_state = hidden_state[indices]
                selected_hidden_state_list.append(selected_hidden_state)

            losses, logits_list = list(), list()
            for i, (selected_hidden_state, label) in enumerate(zip(selected_hidden_state_list, labels)):
                num_sents, hidden_size = selected_hidden_state.size()
                concatenated_hidden_state = torch.zeros(num_sents, num_sents, hidden_size*3, device=selected_hidden_state.device)
                concatenated_hidden_state[:, :, :hidden_size] = selected_hidden_state.unsqueeze(1)
                concatenated_hidden_state[:, :, hidden_size:hidden_size*2] = selected_hidden_state.unsqueeze(0)
                concatenated_hidden_state[:, :, hidden_size*2:hidden_size*3] = torch.abs(selected_hidden_state.unsqueeze(0) - selected_hidden_state.unsqueeze(1))
                logits = self.sim_head(self.dropout(concatenated_hidden_state)).squeeze()

                logits = nn.Sigmoid()(logits)
                real_labels = torch.zeros_like(logits)
                if label.numel():
                    real_labels[label[:, 0], label[:, 1]] = 1
                real_labels += torch.eye(len(logits), device=logits.device)
                loss = nn.MSELoss()(real_labels, logits) + nn.MSELoss()(logits, logits.transpose(0, 1))
                losses.append(loss)

                logits_list.append(logits)
            loss = torch.mean(torch.stack(losses))
            return MaskedLMOutput(loss=loss, logits=logits_list, hidden_states=selected_hidden_state_list)


class SpeakerIdentificationDataset:
    def __init__(self, base_folder, bos_token='<bos>', split='train', dataset='friends', debug=False):
        self.base_folder = base_folder
        self.debug = debug
        self.dataset = dataset
        self.split = split
        self.bos_token = bos_token

        if dataset == 'friends':
            if split == 'test':
                metadata = json.load(open(os.path.join(base_folder, 'test-metadata.json')))
            else:
                metadata = json.load(open(os.path.join(base_folder, 'train-metadata.json')))

            self.examples = list()
            for dialog_data in metadata:
                # use season 01 as valid set
                if split == 'valid' and not dialog_data[0]['frame'].startswith('s01'):
                    continue
                if split == 'train' and dialog_data[0]['frame'].startswith('s01'):
                    continue
                self.examples.append(dialog_data)
        else:
            if dataset == 'ubuntu_dialogue_corpus':
                self.examples = [json.loads(line) for line in open(os.path.join(base_folder, '%s.json' % (split.replace('valid', 'dev'))))]
            self.examples = [example for example in self.examples if len(example['ctx_spk']) != len(set(example['ctx_spk']))]

        print('loaded %d examples' % len(self))
        print('example data:', self[0])

    def __len__(self):
        return len(self.examples) if not self.debug else 32

    def __getitem__(self, index):
        example = self.examples[index]
        if self.dataset == 'friends':
            speakers, contents, frame_names = [i['speaker'] for i in example], [i['content'] for i in example], [i['frame'] for i in example]
        else:
            speakers, contents = example['ctx_spk'], example['context']
            frame_names = ['%d-%d' % (index, i) for i in range(len(speakers))]

        labels = list()
        for i, speaker_i in enumerate(speakers):
            for j, speaker_j in enumerate(speakers):
                if i != j and speaker_i == speaker_j:
                    labels.append([i, j])
        input_text = self.bos_token + self.bos_token.join(contents)
        return input_text, labels, frame_names


class Collator:
    def __init__(self, tokenizer, max_length=512, temperature=1.0, use_turn_emb=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.use_turn_emb = use_turn_emb
        self.print_debug = True

    def __call__(self, examples):
        input_texts = [i[0] for i in examples]
        labels = [i[1] for i in examples]
        frame_names = [i[2] for i in examples]
        model_inputs = self.tokenizer(input_texts, add_special_tokens=False, truncation=True, padding='longest', max_length=self.max_length, return_tensors='pt')
        model_inputs = dict(model_inputs)

        new_labels = list()
        for input_id, label in zip(model_inputs['input_ids'], labels):
            num_bos_tokens = torch.sum(input_id == self.tokenizer.bos_token_id).item()
            label = [l for l in label if l[0] < num_bos_tokens and l[1] < num_bos_tokens]      # remove the truncated turns
            new_labels.append(torch.tensor(label))

        model_inputs['labels'] = new_labels
        if self.use_turn_emb:
            model_inputs['token_type_ids'] = torch.cumsum(model_inputs['input_ids'] == self.tokenizer.bos_token_id, dim=1)
        model_inputs['frame_names'] = frame_names
        model_inputs['temperature'] = self.temperature

        if self.print_debug:
            print(model_inputs)
            self.print_debug = False

        return model_inputs


@torch.no_grad()
def evaluate(dataloader, model):
    model.eval()
    preds_list, golds_list, losses_list = list(), list(), list()
    logits_list, labels_list = list(), list()
    for batch in dataloader:
        batch = to_device(batch, device)
        model_output = model(**batch)
        losses_list.append(model_output.loss.cpu().item())

        for label, logit in zip(batch['labels'], model_output.logits):
            label = {(i, j) for i, j in label.cpu().tolist() if i < j}
            logit = logit[:, :len(logit)].cpu().tolist()
            for i in range(len(logit)):
                for j in range(i+1, len(logit)):
                    preds_list.append(logit[i][j])
                    golds_list.append((i, j) in label)
        logits_list.extend(to_device(model_output.logits, 'cpu'))
        labels_list.extend(to_device(batch['labels'], 'cpu'))

    print('example preds:', preds_list[:50])
    print('example golds:', golds_list[:50])

    loss = np.mean(losses_list)
    auc = roc_auc_score(golds_list, preds_list)
    model.train()
    return loss, auc, {'logits': logits_list, 'labels': labels_list}


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='friends', choices=['friends', 'ubuntu_dialogue_corpus'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deberta_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--turn_emb', action='store_true')
    parser.add_argument('--eval_at_start', action='store_true')

    parser.add_argument('--data_base_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--max_length', type=int, default=512)

    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='set weight decay if you need, especially training on small datasets like friends-mmsi')
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=2000)

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    if args.debug:
        args.log_steps = 2
        args.eval_steps = 8
        args.num_epochs = 40

    tokenizer = AutoTokenizer.from_pretrained(args.deberta_model)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.bos_token_id = tokenizer.cls_token_id

    if args.func == 'train':
        writer = SummaryWriter(args.output_path)
        config = AutoConfig.from_pretrained(args.deberta_model)
        if args.turn_emb:
            config.type_vocab_size = 10     # max history length

        config.bos_token_id = tokenizer.bos_token_id
        print(config.bos_token_id)
        model = DebertaV3ForSpeakerLabeling.from_pretrained(
            args.deberta_model, config=config, ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
        )

        print('loading model from: %s' % args.deberta_model)
        model.to(device)
        train_dataset = SpeakerIdentificationDataset(args.data_base_folder, bos_token=tokenizer.bos_token, dataset=args.dataset, split='train', debug=args.debug)
        valid_dataset = SpeakerIdentificationDataset(args.data_base_folder, bos_token=tokenizer.bos_token, dataset=args.dataset, split='valid' if not args.debug else 'train', debug=args.debug)
        collator = Collator(tokenizer, max_length=args.max_length, temperature=args.temperature, use_turn_emb=args.turn_emb)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

        if args.weight_decay != 0:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            print('weight decay %.4f, params:' % args.weight_decay, [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],)
            print('no weight decay, params:', [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],)
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * args.num_epochs
        )

        global_steps = 0
        best_auc = {'valid': 0., 'test': 0.}

        if args.eval_at_start:
            dataloaders = {'valid': valid_dataloader}
            for split, dataloader in dataloaders.items():
                loss, auc, results = evaluate(dataloader, model)
                print('eval_at_start auc: %.4f' % (auc))
                writer.add_scalar('%s/loss' % split, loss, global_steps)
                writer.add_scalar('%s/auc' % split, auc, global_steps)
                pickle.dump(results, open(os.path.join(args.output_path, '%s_output-eval_at_start.pkl' % split), 'wb'))

        for epoch_i in range(args.num_epochs):
            model.train()
            for batch in train_dataloader:
                global_steps += 1
                batch = to_device(batch, device)
                model_output = model(**batch)
                loss = model_output.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_steps % args.log_steps == 0:
                    writer.add_scalar('train/loss', loss.cpu().item(), global_steps)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_steps)

                if global_steps % args.eval_steps == 0:
                    dataloaders = {'valid': valid_dataloader}
                    for split, dataloader in dataloaders.items():
                        loss, auc, results = evaluate(dataloader, model)
                        writer.add_scalar('%s/loss' % split, loss, global_steps)
                        writer.add_scalar('%s/auc' % split, auc, global_steps)
                        print('auc: %.4f, best_auc: %.4f' % (auc, best_auc[split]))
                        if auc > best_auc[split]:
                            best_auc[split] = auc
                            pickle.dump(results, open(os.path.join(args.output_path, '%s_output.pkl' % split), 'wb'))
                            ckpt_folder = os.path.join(args.output_path, 'checkpoint-%s' % split)
                            model.save_pretrained(ckpt_folder)

    elif args.func == 'test':
        model = DebertaV3ForSpeakerLabeling.from_pretrained(args.deberta_model)
        print('loading model from: %s' % args.deberta_model)
        model.to(device)
        test_dataset = SpeakerIdentificationDataset(args.data_base_folder, bos_token=tokenizer.bos_token, dataset=args.dataset, split='test' if not args.debug else 'train', debug=args.debug)
        collator = Collator(tokenizer, max_length=args.max_length, temperature=args.temperature)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
        loss, auc, results = evaluate(test_dataloader, model)
        print('auc: %.4f' % (auc))
        pickle.dump(results, open(os.path.join(args.output_path, 'test_output.pkl'), 'wb'))

    else:
        raise NotImplementedError()
