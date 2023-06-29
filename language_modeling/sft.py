import random

from datasets import Dataset
from pprint import pprint
from typing import List, Optional
from evaluation.preprocessing.utils import *
import pytorch_lightning as pl
import torch
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    BloomForCausalLM,
    BloomTokenizerFast
)
from evaluation.generation.main import evaluate


def prompt_input(question: str, knowledge_sources: List[any], answer: str = ''):
    knowledge = ''
    for i, ks in enumerate(knowledge_sources):
        knowledge += f"[{i + 1}] {ks['title']}\n{ks['text']}\n"

    out = f"""Given the following question, some extracted knowledge from the web, write an answer with references:
QUESTION: {question}
KNOWLEDGE:
{knowledge}
ANSWER: {answer}
"""
    out = out.replace('\t', '')
    return out


class SFTDataset:
    def __init__(self,
                 items: List[any],
                 tokenizer: AutoTokenizer,
                 max_token_length: int = 512):
        self.tokenizer = tokenizer
        self.items = items
        self.max_token_length = max_token_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]

        # source = prompt_input(item['question'], item['sources'])
        # target = item['answer']
        #
        # source_encoding = self.tokenizer.encode_plus(source,
        #                                         max_length=self.source_max_token_length,
        #                                         padding='max_length',
        #                                         truncation=True,
        #                                         return_attention_mask=True,
        #                                         add_special_tokens=True,
        #                                         return_tensors='pt',
        #                                         # add_prefix_space=True,
        #                                         )
        #
        # target_encoding = self.tokenizer.encode_plus(target,
        #                                         max_length=self.target_max_token_length,
        #                                         padding='max_length',
        #                                         truncation=True,
        #                                         return_attention_mask=True,
        #                                         add_special_tokens=True,
        #                                         return_tensors='pt',
        #                                         # add_prefix_space=True,
        #                                         )
        # print()
        # return dict(
        #     source=source,
        #     target=target,
        #     input_ids=source_encoding['input_ids'].flatten(),
        #     attention_mask=source_encoding['attention_mask'].flatten(),
        #     labels=target_encoding['input_ids'].flatten()
        # )

        prompt = prompt_input(item['question'], item['sources'], item['answer'])
        prompt_encoding = self.tokenizer.encode_plus(prompt,
                                                     max_length=self.max_token_length,
                                                     padding='max_length',
                                                     truncation=True,
                                                     return_attention_mask=True,
                                                     add_special_tokens=True,
                                                     return_tensors='pt',
                                                     # add_prefix_space=True,
                                                     )
        return dict(
            prompt=prompt,
            input_ids=prompt_encoding['input_ids'].flatten(),
            attention_mask=prompt_encoding['attention_mask'].flatten(),
            labels=prompt_encoding['input_ids'].flatten()
        )


class SFTDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: List[any],
                 val_data: List[any],
                 test_data: List[any],
                 tokenizer: AutoTokenizer,
                 batch_size: int = 8,
                 max_token_length: int = 512,
                 num_workers: int = 4
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self, *args, **kwargs):
        # only called within a single process, global_rank == 0
        pass

    def setup(self, stage: Optional[str] = None):
        # called on every process when using DDP
        self.train_dataset = SFTDataset(self.train_data,
                                        self.tokenizer,
                                        self.max_token_length)
        self.val_dataset = SFTDataset(self.val_data,
                                      self.tokenizer,
                                      self.max_token_length)
        self.test_dataset = SFTDataset(self.test_data,
                                       self.tokenizer,
                                       self.max_token_length)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=False, num_workers=1)


class BloomSFTModel(pl.LightningModule):
    def __init__(self):
        super(BloomSFTModel, self).__init__()
        self.save_hyperparameters()
        model_path = '/nfs-data/models/bloom-560m'
        self.lr = 2e-5
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BloomForCausalLM.from_pretrained(model_path, return_dict=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss


def map_webgpt_comparison_items_to_perplexity_format(webgpt_items):
    results = []
    for item1, item2 in webgpt_items:
        target_item = item1 if item1['score'] > item2['score'] else item2
        identifier = target_item['question']['id']
        question = target_item['question']['full_text']
        sources = []
        for q in target_item['quotes']:
            sources.append({
                'title': q['title'],
                'text': q['extract']
            })
        answer = target_item['answer']
        results.append({
            'id': identifier,
            'question': question,
            'sources': sources,
            'answer': answer
        })
    return results


def do_training():
    # lr = 2e-5
    batch_size = 32
    max_epochs = 100
    val_per_epoch = 5

    cwd = os.getcwd()
    print('current directory: ', cwd)
    ckpt_dir = os.path.join(cwd, 'checkpoints')
    logs_dir = os.path.join(cwd, 'logs')
    clear_directory(ckpt_dir)
    clear_directory(logs_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    logger = TensorBoardLogger(
        save_dir=cwd,
        version=1,
        name='logs'
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
        val_check_interval=1.0 / val_per_epoch,
        # gpus=[0, 1, 2, 3],
        # gpus=[0, 1, 2, 3, 4, 5, 6, 7],
        gpus=[4, 5, 6, 7],
        strategy='dp',
        progress_bar_refresh_rate=1,
        logger=logger
    )
    pl.seed_everything(42)  # for reproducibility
    model = BloomSFTModel()
    train_set1 = load_jsonl('/nfs-data/datasets/perplexity_ai/eli5/outputs_0_54014_train.jsonl')
    val_set1 = load_jsonl('/nfs-data/datasets/perplexity_ai/eli5/outputs_0_54014_val.jsonl')
    test_set1 = load_jsonl('/nfs-data/datasets/perplexity_ai/eli5/outputs_0_54014_test.jsonl')
    train_set = load_jsonl('/nfs-data/datasets/webgpt/comparisons_filtered_train.jsonl')
    train_set = map_webgpt_comparison_items_to_perplexity_format(train_set) + train_set1
    val_set = load_jsonl('/nfs-data/datasets/webgpt/comparisons_filtered_val.jsonl')
    val_set = map_webgpt_comparison_items_to_perplexity_format(val_set) + val_set1
    test_set = load_jsonl('/nfs-data/datasets/webgpt/comparisons_filtered_test.jsonl')
    test_set = map_webgpt_comparison_items_to_perplexity_format(test_set) + test_set1
    random.shuffle(train_set)
    random.shuffle(val_set)
    print(f'train set: {len(train_set)}')
    print(f'val set: {len(val_set)}')
    print()
    datamodule = SFTDataModule(tokenizer=model.tokenizer,
                               train_data=train_set,
                               val_data=val_set,
                               test_data=test_set,
                               batch_size=batch_size)
    # training
    trainer.fit(model, datamodule)
    print()


def extract_answer(generated_response: str) -> str:
    answer = generated_response.split('\nANSWER: ')[1]
    answer = answer.split('"""')[0].replace('\n', '')
    answer = trim_incomplete_sentence(answer)
    return answer


def save_hf_ckpt(model, tokenizer, target_dir):
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    print(f'model and tokenizer saved to {target_dir}')


def do_prediction():
    device = 'cuda:3'
    data_path = '/nfs-data/datasets/perplexity_ai/eli5/outputs_0_54014_test.jsonl'
    items = load_jsonl(data_path, max_line_count=500)

    ckpt_path = '/nfs-data/bo/chair/training/checkpoints/best-checkpoint.ckpt'
    ckpt = BloomSFTModel.load_from_checkpoint(ckpt_path)
    ckpt.eval()
    model = ckpt.model
    tokenizer = ckpt.tokenizer

    # model_path = '/nfs-data/bo/checkpoints/bloom-560m-sft-perplexity-default-padding'
    # model = BloomForCausalLM.from_pretrained(model_path)
    # tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    model.eval()
    model.to(device)
    gen_args = {
        "max_new_tokens": 128,
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.92,
        "temperature": 0.8,
    }
    golds = [item['answer'] for item in items]
    preds = []
    for batch in yield_batches(items, batch_size=32):
        prompts = [prompt_input(item['question'], item['sources']) for item in batch]
        inputs = tokenizer.batch_encode_plus(prompts,
                                             padding='longest',
                                             return_attention_mask=True,
                                             add_special_tokens=True,
                                             return_tensors='pt')
        inputs.to(device)
        outputs = model.generate(**inputs, **gen_args)
        responses = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        answers = [extract_answer(r) for r in responses]
        preds = preds + answers
        print(inputs['input_ids'].shape)

    result = evaluate(preds, golds, device=device)['scores_meta']
    pprint(result)
    print()


if __name__ == '__main__':
    print('start')
    # do_training()
    do_prediction()
    print('end')

