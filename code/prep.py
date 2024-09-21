r"""
Author: XUE Boyang      Filename: prep.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Data preparation: parse, preprocess, and save. 
Data output format: 
    {
        "question_id": "question_id",
        "question": "question",
        "answer": "answer"
    }
"""
import os
import argparse
import json

import datasets
import pandas as pd

from utils import read_json, read_jsonl, write_json, write_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="webqa", choices=["triviaqa", "webqa", "gsm8k"])
parser.add_argument('--output_dir', type=str, default="./data/{}")
args = parser.parse_args()


"""
TriviaQA dataset preparation and saving
"""
def prep_triviaqa_dataset(split="validation"):
    print(f'Preprocessing TriviaQA {split} dataset')
    data_pool = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
    id_mem = set()

    def remove_dups(batch):
        if batch['question_id'][0] in id_mem:
            return {_:[] for _ in batch.keys()}
        id_mem.add(batch['question_id'][0])

        return batch

    data_pool = data_pool.map(remove_dups, batch_size=1, batched=True, 
                            load_from_cache_file=False, remove_columns=["search_results", "question_source", "entity_pages"])

    # Warrant the duplicated data was removed
    assert pd.Series([_['question_id'] for _ in data_pool]).value_counts().max() == 1

    data_set = []
    for data in data_pool:
        # import pdb; pdb.set_trace()
        instance = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"]["value"]
        }
        data_set.append(instance)
    
    print(f"Data size of {split}: {len(data_set)}")

    return data_set


def get_triviaqa_dataset(output_dir):
    # Get data splits
    data_splits = ["train", "validation", "test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_triviaqa_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
WebQA dataset preparation and saving
"""
def prep_webqa_dataset(data_path, split="validation"):
    print(f'Preprocessing WebQA {split} dataset')
    data_pool = json.load(open(data_path, "r"))
    print(f"Original data size of {split}: {len(data_pool)}")

    # import pdb; pdb.set_trace()
    data_set = []
    for key, value in data_pool.items():
        answers = []
        for answer in value["evidences"].values():
            if answer["answer"][0] != "no_answer":
                # import pdb; pdb.set_trace()
                answers.append(answer["answer"][0])

        # import pdb; pdb.set_trace()
        # print(answers)
        if answers:
            instance = {
                "question_id": key,
                "question": value["question"],
                "answer": max(answers, key=answers.count)
            }
            data_set.append(instance)
    
    print(f"Processed data size of {split}: {len(data_set)}")

    return data_set


def get_webqa_dataset(output_dir):
    # Get data splits
    data_splits = ["me_train", "me_validation.ir", "me_validation.ann", "me_test.ir", "me_test.ann"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_path = os.path.join("./../../data/WebQA.v1.0", f"{split}.json")
        data_set = prep_webqa_dataset(data_path, split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
GSM8K dataset preparation and saving
"""
def prep_gsm8k_dataset(split="test"):
    print(f'Preprocessing GSM8K {split} dataset')
    # Load GSM8K dataset
    dataset = datasets.load_dataset("gsm8k", "main", split=split)

    # data_set data questions and answers
    data_set = []
    for idx, item in enumerate(dataset):
        data_set.append({
            "question_id": idx,
            "question": item['question'],
            "answer": item['answer'] + "<|endoftext|>"
        })

    # Examples to construct few-shot examples
    if split == "train":
        examplars = {}
        for idx in range(0, 80, 8):
            examplars[f"examplar_{int(idx/8)+1}"] = data_set[idx:idx+8]
        
        prompt_path = "./data/gsm8k/few_shot_examplars.json"
        write_json(prompt_path, examplars)

    print(f"{len(data_set)} {split} examples")

    return data_set


def get_gsm8k_dataset(output_dir):
    # Get data splits
    data_splits = ["train", "test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_gsm8k_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


if __name__=="__main__":
    # Output directory
    output_dir = args.output_dir.format(args.dataset)
    print(f"Data saved to {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset == "trivia_qa":
        get_triviaqa_dataset(output_dir)
    elif args.dataset == "web_qa":
        get_webqa_dataset(output_dir)
    elif args.dataset == "gsm8k":
        get_gsm8k_dataset(output_dir)

