import os
import hydra
from pprint import pprint
from omegaconf import DictConfig, OmegaConf
from evaluation.metrics import TextMetrics, BertMetrics, F1Metrics
from evaluation.preprocessing.utils import *
from tqdm import tqdm
from typing import List, Dict, Union
import numpy as np
import pytorch_lightning as pl
from evaluation.generation.answer_generation import (
   load_bloom_model_for_inference,
   chat,
   mock_retrieve_documents_with_perplexity_datum,
)


def read_config() -> DictConfig:
   hydra.initialize(config_path="./", version_base="1.2.0")
   cfg = hydra.compose(config_name="config")
   print(cfg)
   return cfg


def evaluate(preds: List[str], golds: List[str], device: str = 'cuda:0') -> dict:
   tm = TextMetrics()
   bm = BertMetrics(device=device)
   all_scores = []
   for pred, gold in tqdm(zip(preds, golds), total=len(golds)):
       scores = tm.measure(pred, gold)
       bert_score = bm.measure(pred, gold)
       scores.update({"bert": bert_score})
       all_scores.append(scores)

   def mean_score(key: str, arr: List[any]):
       scores = [r[key] for r in arr]
       mean_s, sd_s = float(np.mean(scores)), float(np.std(scores))
       # print(f"{key}: mean={round(mean_s, 3)}, sd={round(sd_s, 3)} ")
       return mean_s, sd_s

   keys = ["bleu1", "bleu2", "bleu4", "rouge1", "rouge2", "rougeL", "mer", "bert"]
   scores_meta = {}
   for k in keys:
       m, s = mean_score(k, all_scores)
       scores_meta.update({k: {"mean": m, "sd": s}})

   return {
       'scores': all_scores,
       'scores_meta': scores_meta
   }


def evaluate_sft(cfg: DictConfig):
   output_dir = cfg["output_dir"]
   model_path = cfg["model_path"]
   data_path = cfg["data_path"]
   seed = cfg["seed"]
   device = cfg['device']
   gen_mode = cfg["generation_mode"]
   gen_args = cfg["generation_args"][gen_mode]
   mnt = gen_args["max_new_tokens"]
   config_obj = OmegaConf.to_object(cfg)
   pl.seed_everything(seed)
   output_path = os.path.join(
       output_dir, f"eval-bloom1b-sft-{gen_mode}-max{mnt}tokens.json"
   )
   model, tokenizer = load_bloom_model_for_inference(model_path, device=device)

   data = load_jsonl(data_path, max_line_count=3)
   preds =[]
   answers = []
   answers_detailed = []
   results = []
   for item in tqdm(data):
       question = item["question"]
       answer = item["answer"]
       answer_detailed = item["answer_detailed"]
       answers.append(answer)
       answers_detailed.append(answer_detailed)
       docs = mock_retrieve_documents_with_perplexity_datum(item)
       pred_with_mock_docs = chat(model, tokenizer, question, docs, gen_args=gen_args)
       print(pred_with_mock_docs)
       preds.append(pred_with_mock_docs)

   e1 = evaluate(preds, answers)
   scores_answers, meta1 = e1['scores'], e1['scores_meta']
   e2 = evaluate(preds, answers_detailed)
   scores_answers_detailed, meta2 = e2['scores'], e2['scores_meta']
   for i, item in enumerate(data):
       item.update(
           {
               "answer_generated": preds[i],
               "scores_answer": scores_answers[i],
               "scores_answer_detailed": scores_answers_detailed[i],
           }
       )
       results.append(item)

   results_dict: Dict[str, Union[str, List, Dict]] = {
       "config": config_obj,
       "results": results,
       "num_examples": len(data),
       "scores_answer": meta1,
       "scores_answer_detailed": meta2,
   }
   print('answers scores:')
   pprint(meta1)
   print('answers_detailed scores:')
   pprint(meta2)
   # save_json(output_path, results_dict)
   print()


@hydra.main(config_path="./", config_name="config", version_base="1.2.0")
def evaluate_sft_sweep(cfg: DictConfig) -> None:
   for mnt in [128, 256]:
       for gm in ["beam", "top_k", "top_p"]:
           cfg["generation_mode"] = gm
           cfg["generation_args"][gm]["max_new_tokens"] = mnt
           print(cfg)

   print()


if __name__ == "__main__":
   # evaluate_sft_sweep()
   cfg = read_config()
   evaluate_sft(cfg)

