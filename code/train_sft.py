r"""
Author: XUE Boyang      Filename: train_sft.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Supervised fine-tuning on QA dataset.
"""
import os
import json
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names, print_trainable_parameters

# Constants
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

model_path_dict = {
    "llama3": "/workspace/model/Meta-Llama-3.1-8B-Instruct"
}


@dataclass
class ModelArguments:
    model_name: str = field(default="llama3", metadata={"help": "Model name.", "choices": ["llama3", "vicuna"]})
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})

@dataclass
class DataArguments:
    data_dir: str = field(default="./data/{}/prep", metadata={"help": "Directory to save data."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": ["triviaqa", "webqa", "fast"]})
    data_file: str = field(default="train_2w", metadata={"help": "Data file name."})
    prompt_dir: str = field(default="./prompt/", metadata={"help": "Path to the prompt."})
    continue_generate: bool = field(default=False, metadata={"help": "Continue from the previous generations."})
    save_suffix: str = field(default="test", metadata={"help": "File name to save the results."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    output_dir: str = field(default="./exp/{}/", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing to save memory."})
    max_grad_norm: float = field(default=0.3, metadata={"help": "Max gradient norm."})
    num_train_epochs: int = field(default=5, metadata={"help": "Total number of training epochs to perform."})
    learning_rate: float = field(default=2e-4, metadata={"help": "The initial learning rate for Adam."})
    fp16: bool = field(default=True, metadata={"help": "Whether to use fp16 for training."})
    bf16: bool = field(default=False, metadata={"help": "Whether to use bfloat16 for training."})
    save_total_limit: int = field(default=3, metadata={"help": "Limit the total amount of checkpoints."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    optim: str = field(default="paged_adamw_32bit", metadata={"help": "Optimizer to use."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type."})
    warmup_ratio: float = field(default=0.05, metadata={"help": "Warmup ratio."})


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


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up logging.
    training_args.output_dir = os.path.join(training_args.output_dir.format(data_args.dataset), 
                                         f"{model_args.model_name}_{data_args.save_suffix}")
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    training_args.log_path = os.path.join(training_args.output_dir, f"train.log")

    logging.basicConfig(
        filename=training_args.log_path,
        filemode='w',
        level=logging.INFO,
        datefmt="%d-%M-%Y %H:%M:%S",
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Load model and tokenizer
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_name_or_path = model_path_dict[model_args.model_name]
    logging.info(f"Loading model and tokenizer from {model_name_or_path} ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                            torch_dtype=torch.float16, 
                                                            quantization_config=bnb_config,
                                                            device_map="balanced")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    # import pdb; pdb.set_trace()
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

    # import pdb; pdb.set_trace()
    # Load prompt and select the prompt type.
    if data_args.dataset == "fast":
        instruction = "Answer the following question."
        prompt_input = "{}\n{}"
    else:
        prompt_template = json.load(open(os.path.join(data_args.prompt_dir, 
                                                    f"{data_args.dataset}_template.json")))
        instruction = prompt_template["instruction"]
        prompt_input = prompt_template["standard_prompt"]

    # Load dataset
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset), f"{data_args.data_file}.json")
    logging.info(f"Loading data from {data_path} ...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    # dataset = load_dataset("json", data_files="./data/fast/conversations.json",split="train")
    print("Number of training samples: ", len(dataset))

    # Save path
    output_dir = training_args.output_dir.format(data_args.dataset)

    def formatting_prompts_func(example):
        output_texts = []
        # print(example)
        for idx in range(len(example['question'])):
            text = prompt_input.format(instruction, example['question'][idx]) + example['answer'][idx]
            output_texts.append(text)
        # print(output_texts)
        return output_texts

    # Parameters for training arguments details
    training_args = TrainingArguments(
        output_dir=output_dir
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,
        formatting_func=formatting_prompts_func,
        args=training_args
    )

    trainer.train() 
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

