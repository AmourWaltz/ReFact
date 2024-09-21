# ReFact: Reinforcing Factuality for LLMs




## Implementation

### Stage 1: Data construction and split the difficulty levels

Dataset Sampling and Construction

```shell
# model: llama3 qwen2
# dataset: gsm8k, triviaqa - llama3
#          webqa - qwen2
# data_file: 

python code/sample.py  --model_name llama3 --dataset gsm8k --data_file test --suffix sample_8shot

python code/split.py --model_name llama3 --dataset gsm8k --data_file test --suffix sample_8shot
```

### SFT: Supervised Fine-tune

```shell
if [ $stage == train ]; then
  echo "$0: Fine-tune Alpaca on $data data set."
    torchrun --nproc_per_node=4 --master_port=1234 code/align/train_sft.py \
        --model_name_or_path models/alpaca \
        --data_path ./data/${data}.json \
        --prompt_path ./templates/${prompt}.json \
        --bf16 True \
        --output_dir models/$model \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True

fi
```


### Stage 3: LLM inference and evaluation part.

```shell
# TriviaQA
python code/infer.py --model_name llama3 --dataset triviaqa --data_file validation --suffix sample_2k_1shot --max_len 16

# GSM8K
python code/infer.py --model_name llama3 --dataset gsm8k --suffix sample_8shot --data_file test --max_len 1024

# TriviaQA
python code/infer.py --model_name qwen2 --dataset webqa --data_file validation --suffix sample_2k_1shot --max_len 16
```



### Stage 3: Evaluate the generations

```shell
# TriviaQA
python code/eval.py --model_name llama3 --dataset triviaqa --data_file validation --suffix sample_2k_1shot

```



### Stage 4: Generate related knowledge based on questions by Alpaca.

```shell
if [ $stage == gen ]; then
  echo "$0: Generate related knowledge based on questions by Alpaca."
    CUDA_VISIBLE_DEVICES=0
    torchrun --nproc_per_node=1 --master_port=25678 codes/$stage.py \
        --model_name_or_path ./models/$model \
        --data_path ./data/train_$data.json \
        --prompt_path ./templates/${prompt}.json \
        --instruction_path ./instructions/${prompt}.txt \
        --output_dir ./dev/train_${prompt} \
        --temperature 0.1 \
        --top_p 0.9 \
        --top_k 20 \
        --num_beams 4


fi
```

