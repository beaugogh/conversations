import os
import shutil
from time import time
import datetime
import json
import math
from typing import List, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def load_json(data_path: str):
   print(f"\nloading json from {data_path}...")
   start = time()
   with open(data_path, "r") as file_reader:
       data = json.load(file_reader)
       print("json is loaded from ", data_path)
       lapse = time() - start
       print(f"time elapsed: {datetime.timedelta(seconds=lapse)}")
       return data


def save_json(data_path: str, target_object: any):
   print(f"\nsaving json to {data_path}...")
   start = time()
   with open(data_path, "w", encoding="utf-8") as file_writer:
       json.dump(target_object, file_writer, ensure_ascii=False)
       print("json is saved to ", data_path)
       lapse = time() - start
       print(f"time elapsed: {datetime.timedelta(seconds=lapse)}")


def load_jsonl(data_path: str, max_line_count: int = math.inf) -> List[any]:
   data = []
   count = 0
   start = time()
   with open(data_path, "r") as f:
       for l in tqdm(f):
           count += 1
           if count <= max_line_count:
               data.append(json.loads(l))
           else:
               break

       print("jsonl is loaded from ", data_path)
       lapse = time() - start
       print(f"time elapsed: {datetime.timedelta(seconds=lapse)}")
       return data


def save_jsonl(data_path: str, target_list: List[any]):
   assert ".jsonl" in data_path
   print(f"\nsaving jsonl to {data_path}...")
   start = time()
   with open(data_path, "w", encoding="utf-8") as f:
       for obj in tqdm(target_list):
           line = json.dumps(obj, ensure_ascii=False)
           f.write(line + "\n")

       print("jsonl is saved to ", data_path)
       lapse = time() - start
       print(f"time elapsed: {datetime.timedelta(seconds=lapse)}")


def split_train_val_test(
       items: List[any], first_split: float = 0.1, second_split: float = 0.5
):
   train_set, remaining = train_test_split(items, test_size=first_split, shuffle=True)
   val_set, test_set = train_test_split(items, test_size=second_split, shuffle=True)
   return train_set, val_set, test_set


def save_splits(
       output_path_prefix: str,
       train_data: List[any] or Dict,
       val_data: List[any] or Dict,
       test_data: List[any] or Dict,
):
   assert (
           ".jsonl" in output_path_prefix or ".json" in output_path_prefix
   ), "only support saving jsonl or json"
   suffix = ".json"
   save_func = save_json
   if ".jsonl" in output_path_prefix:
       suffix = ".jsonl"
       save_func = save_jsonl

   save_func(output_path_prefix.replace(suffix, f"_train{suffix}"), train_data)
   save_func(output_path_prefix.replace(suffix, f"_val{suffix}"), val_data)
   save_func(output_path_prefix.replace(suffix, f"_test{suffix}"), test_data)


def timestamp():
   time_stamp = str(datetime.datetime.now())
   time_stamp = time_stamp.replace(" ", "_").replace(":", "-").split(".")[0]
   return time_stamp


def clear_directory(folder_path: str):
   if not os.path.isdir(folder_path):
       os.mkdir(folder_path)
   for filename in os.listdir(folder_path):
       file_path = os.path.join(folder_path, filename)
       try:
           if os.path.isfile(file_path) or os.path.islink(file_path):
               os.unlink(file_path)
           elif os.path.isdir(file_path):
               shutil.rmtree(file_path)
           print(f'Successfully cleared {file_path}')
       except Exception as e:
           print('Failed to delete %s. Reason: %s' % (file_path, e))


def yield_batches(items: List[any], batch_size: int = 12):
   for i in tqdm(range(0, len(items), batch_size)):
       yield items[i: i + batch_size]


def trim_incomplete_sentence(sentence: str) -> str:
   sent = sentence + ""
   if sent:
       last = sent[-1]
       while sent and last not in [".", "!", "]"]:
           sent = sent[:-1]
           if len(sent) == 0:
               break
           last = sent[-1]

   return sent

