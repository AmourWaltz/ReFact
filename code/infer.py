r"""
Author: XUE Boyang      Filename: infer.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Inference decoding on validation set.
"""
import os
import sys
import time
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from tqdm import tqdm
import json
import random

import torch
import transformers
from transformers import GenerationConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from utils import format_seconds, read_jsonl, jsonl2json, model_path_dict

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = torch.device("cuda")


@dataclass
class ModelArguments:
    model_name: str = field(default="llama3", metadata={"help": "Model name.", "choices": ["llama3", "qwen2"]})
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})


@dataclass
class DataArguments:
    data_dir: str = field(default="./data/{}/prep", metadata={"help": "Directory to save data."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": ["triviaqa", "webqa", "gsm8k"]})
    data_file: str = field(default="validation", metadata={"help": "Data file name."})
    prompt_dir: str = field(default="./prompt/", metadata={"help": "Path to the prompt."})
    continue_generate: bool = field(default=False, metadata={"help": "Continue from the previous generations."})


@dataclass
class InferenceArguments:
    icl_use: bool = field(default=True, metadata={"help": "Use few-shot prompt or not."})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling or not."})
    output_dir: str = field(default="./exp/{}/infer", metadata={"help": "Directory to save results."})
    suffix: str = field(default="vanilla", metadata={"help": "File name to save the results."})
    num_sampling: int = field(default=5, metadata={"help": "Number of samples."})
    temperature: float = field(default=0.8, metadata={"help": "Temperature for sampling."})
    top_p: float = field(default=1.0, metadata={"help": "Top p for sampling."})
    top_k: int = field(default=40, metadata={"help": "Top k for sampling."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for sampling."})
    max_length: int = field(default=16, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})


# Resize tokenizer and embedding.
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# Format the few-shot examplar of list to string.
def format_examplar(few_shot_examplars, examplar_split):
    few_shot_examplar_list = []
    for few_shot_examplar in few_shot_examplars.values():
        few_shot_examplas = []
        for few_shot_example in few_shot_examplar:
            few_shot_examplas.append(f"{examplar_split["input"]}{few_shot_example["question"]}\n{examplar_split["output"]}{few_shot_example["answer"]}")
        few_shot_examplar_list.append("\n\n".join(few_shot_examplas))

    return few_shot_examplar_list


# Split the generation to get the answer part.
def output_split(output, tokenizer, split_len, dataset, prompt_split):
    if dataset in ["triviaqa", "webqa"]:
        return tokenizer.decode(output.sequences[0][split_len:], 
                                skip_special_tokens=True).split("###")[0].replace("\n", "").lstrip()
    elif dataset == "gsm8k":
        return tokenizer.decode(output.sequences[0][split_len:], 
                                skip_special_tokens=True).split(prompt_split)[0]


def infer():
    # import pdb; pdb.set_trace()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    model_args, data_args, infer_args = parser.parse_args_into_dataclasses()

    # Set up logging.
    infer_args.output_dir = os.path.join(infer_args.output_dir.format(data_args.dataset), 
                                         f"{model_args.model_name}_{data_args.dataset}_{data_args.data_file}_{infer_args.suffix}")
    if not os.path.exists(infer_args.output_dir):
        os.makedirs(infer_args.output_dir)

    infer_args.log_path = os.path.join(infer_args.output_dir, f"generate.log")

    logging.basicConfig(
        filename=infer_args.log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Load model and tokenizer.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model_name_or_path = model_path_dict[model_args.model_name]
    logging.info(f"Loading model and tokenizer from {model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=torch.float16,
        # quantization_config=bnb_config,
        device_map="balanced" # device_map: "auto", "balanced", "balanced_low_0", "sequential"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Resize tokenizer and embedding.
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token == "":
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token == "":
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token == "":
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # Load data.
    # import pdb; pdb.set_trace()
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset), 
                                  "{}_{}_{}_{}.json".format(model_args.model_name, data_args.dataset, data_args.data_file, infer_args.suffix))
    logging.info(f"Loading data from {data_path} ...")
    dataset = json.load(open(data_path))

    # Load prompt and select the prompt type.
    prompt_template = json.load(open(os.path.join(data_args.prompt_dir, 
                                                  f"{data_args.dataset}_template.json")))
    instruction = prompt_template["instruction"]
    prompt_split = prompt_template["output_split"]
    if infer_args.icl_use:
        few_shot_examplars, examplar_split = prompt_template["generate_few_shot_examplar"], prompt_template["few_shot_split"]
        few_shot_examplar_list = format_examplar(few_shot_examplars, examplar_split)
        prompt_input = prompt_template["few_shot_prompt"]
    else:
        prompt_input = prompt_template["standard_prompt"]


    # Format the output file.
    # import pdb; pdb.set_trace()
    infer_args.save_path = os.path.join(infer_args.output_dir, "generate.json")
    if data_args.continue_generate:
        exist_num = len(read_jsonl(infer_args.save_path))
        # Split the dataset if needed.
        dataset = dataset[exist_num*data_args.sample_interval::]
    else:
        # dataset = dataset[:3]
        open(infer_args.save_path, "w").close()

    data_len = len(dataset)

    logging.info(f"Arguments:\nModel Arguments: {model_args}\nData Arguments: {data_args}\nInference Arguments: {infer_args}")
    logging.info(f"The number of dataset: {data_len}")

    # Sample the data.
    start_time = time.time()
    logging.info("Start generating ...")
    log_flag = True
    with tqdm(total=data_len) as t:
        for idx, data_point in enumerate(dataset):
            # import pdb; pdb.set_trace()
            # time.sleep(1)
            if infer_args.icl_use:
                # import pdb; pdb.set_trace()
                few_shot_examplar = random.choice(few_shot_examplar_list)
                input_ids = tokenizer(prompt_input.format(instruction=instruction,
                                                            examples=few_shot_examplar,
                                                            question=data_point["question"]), 
                                                            return_tensors="pt")["input_ids"].to(device)
            else:
                input_ids = tokenizer(prompt_input.format(instruction=instruction, 
                                                          question=data_point["question"]), 
                                                          return_tensors="pt")["input_ids"].to(device)
                
            with torch.no_grad():
                generation_config = GenerationConfig(
                    do_sample=True,
                    temperature=infer_args.temperature,
                    top_p=infer_args.top_p,
                    top_k=infer_args.top_k,
                    num_beams=infer_args.num_beams,
                    repetition_penalty=1.1
                ) if infer_args.do_sample \
                else GenerationConfig(
                    do_sample=False,
                    num_beams=infer_args.num_beams,
                    repetition_penalty=1.1)

                generation = model.generate(input_ids,
                                        generation_config=generation_config,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        max_new_tokens=infer_args.max_length,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id, 
                                        bos_token_id=tokenizer.bos_token_id)

                # import pdb; pdb.set_trace()
                generation = output_split(generation, tokenizer, len(input_ids[0]), data_args.dataset, prompt_split)

                if log_flag:
                    logging.info(f"Input Prompt: \n{prompt_input.format(instruction=instruction, examples=few_shot_examplar, question=data_point["question"])}")
                    logging.info(f"LLM Generation: \n{generation}")
                    log_flag = False

            # print(data_point)
            instance = {
                "question_id": data_point["question_id"] if "question_id" in data_point.keys() else f"id_{idx+1}",
                "question": data_point["question"],
                "answer": data_point["answer"],
                "output": generation,
                "greedy_scores_avg": data_point["scores"]["greedy_scores_avg"],
                "sample_scores_avg": data_point["scores"]["sample_scores_avg"]
            }
            # print(instance)

            # Real-time saving the results.
            with open(infer_args.save_path, "a+") as fw: 
                instance_write = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(instance_write + '\n')

            t.set_postfix()
            t.update(1)
    
    # import pdb; pdb.set_trace()
    elapsed_time = format_seconds(time.time() - start_time)
    logging.info(f"Total elapsed time: {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s")

    # Convert jsonl to json format.
    logging.info("Generating is done.")
    jsonl2json(infer_args.save_path, infer_args.save_path)
    logging.info(f"Save to {infer_args.save_path}")


if __name__ == "__main__":
    infer()

