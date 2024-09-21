r"""
Author: XUE Boyang      Filename: eval.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Evaluation scripts on QA and Math datasets.
"""
import logging
import re
import os
import string
import argparse

from utils import read_json

from split import known_level, Levels

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="llama3", help="Model name.", choices=["llama3", "qwen2"])
parser.add_argument('--dataset', type=str, default="triviaqa", help="Dataset name.", choices=["triviaqa", "webqa", "gsm8k"])
parser.add_argument('--data_file', type=str, default="validation", help="Data file name.", choices=["validation", "train", # for triviaqa
                                                                       "validation", "me_train", # for webqa
                                                                       "train", "test" # for gsm8k
                    ])
parser.add_argument('--suffix', type=str, default="sample_2k_1shot", help="File name to save the results.")
parser.add_argument('--input_dir', type=str, default="./exp/{}/infer/")

args = parser.parse_args()

"""
Evaluate QA outputs: TriviaQA, WebQA
"""
# Normalize the answer.
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    # return white_space_fix(remove_punc(lower(s)))
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Compute the exact match score in different ways.
def compute_exact(a_gold, a_pred):
    eval_type = "EM_RP"

    if eval_type == "EM":
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))
    elif eval_type == "EM_R":
        return int(normalize_answer(a_gold) in normalize_answer(a_pred))
    elif eval_type == "EM_P":
        return int(normalize_answer(a_pred) in normalize_answer(a_gold))
    elif eval_type == "EM_RP":
        return int(normalize_answer(a_gold) in normalize_answer(a_pred)) or int(normalize_answer(a_pred) in normalize_answer(a_gold))


"""
Evaluate math problems: GSM8K
"""
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example)
    assert gt_answer != INVALID_ANS
    return int(extract_answer(model_completion) == gt_answer)


"""
Evaluate part
"""
def compute_sample_score(label, output, dataset):
    # import pdb; pdb.set_trace()
    if dataset in ["triviaqa", "webqa"]:
        score = compute_exact(label, output)
    elif dataset in ["gsm8k"]:
        # import pdb; pdb.set_trace()
        score = is_correct(output, label)

    return score


def compute_scores(outputs, decoding="greedy", gold_answer=None, dataset=None):
    # For greedy decoding answers
    scores = []
    for output in outputs:
        if decoding == "greedy":
            greedy_answer = output["greedy_decoding"]
            greedy_score = compute_sample_score(gold_answer, greedy_answer, dataset)
            scores.append(greedy_score)
            average_score = sum(scores) / len(scores)
        # For sampling answers
        else:
            sample_scores = []
            # import pdb; pdb.set_trace()
            for sample_output in output["temperature_sampling"]:
                sample_score = compute_sample_score(gold_answer, sample_output, dataset)
                sample_scores.append(sample_score)

            scores.append(sample_scores)
            average_score = sum(map(sum, scores)) / (len(scores) * len(scores[0]))

    return scores, average_score


def evaluate():
    # Format output file.
    args.input_dir = os.path.join(args.input_dir.format(args.dataset), "{}_{}_{}_{}". \
                                   format(args.model_name, args.dataset, args.data_file, args.suffix))
    log_path = os.path.join(args.input_dir, "eval.log")
    if not os.path.exists(args.input_dir):
        os.mkdir(args.input_dir)

    # Loading dataset.
    data_path = os.path.join(args.input_dir, "generate.json")
    data_pool = read_json(data_path)
    
    # print(log_path)
    # Set logging.
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    logging.info(f"Load data from {data_path} ...")

    categories = {}
    categories_correct = {}
    for idx in range(len(Levels)):
        categories[Levels[idx+1]] = 0
        categories_correct[Levels[idx+1]] = 0
        
    for data in data_pool:
        data_known = known_level(data["greedy_scores_avg"], data["sample_scores_avg"], args.dataset)
        categories[data_known] += 1
        score = compute_sample_score(label=data["answer"], output=data["output"], dataset=args.dataset)
        # print(data_known, score)
        categories_correct[data_known] += score

    assert sum(categories.values()) == len(data_pool)

    # logging.info(f"Categories: {categories}")
    # logging.info(f"Categories correct: {categories_correct}")
    logging.info(f"Total Accuracy: {sum(categories_correct.values())/sum(categories.values())}")
    for idx in range(len(Levels)):
        logging.info({
            "Level": Levels[idx+1],
            "Total": categories[Levels[idx+1]],
            "Correct": categories_correct[Levels[idx+1]],
            "Accuracy": round(categories_correct[Levels[idx+1]]/categories[Levels[idx+1]], 4)
            })


if __name__=="__main__":
    evaluate()
