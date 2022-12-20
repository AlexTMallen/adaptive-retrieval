#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from tqdm import tqdm
import json
import argparse
sns.set_theme()


seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('Cuda:', torch.cuda.is_available())
print('pwd', os.getcwd())

from transformers import AutoTokenizer, AutoModelForCausalLM
from util_clm import convert_model_to_int8_on_gpu



import jsonlines

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

q_templates = {
    22: "What is {}'s occupation?",
    218: "In what city was {} born?",
    91: "What genre is {}?",
    257: "Who is the father of {}?",
    182: "In what country is {}?",
    164: "Who was the producer of {}?",
    526: "Who was the director of {}?",
    97: "What is {} the capital of?",
    533: "Who was the screenwriter for {}?",
    639: "Who was the composer of {}?",
    472: "What color is {}?",
    106: "What is the religion of {}?",
    560: "What sport does {} play?",
    484: "Who is the author of {}?",
    292: "Who is the mother of {}?",
    422: "What is the capital of {}?"
}
completion_template = "Q: {} A:"  # "{}" # "Query: {}\nResult:" # "Q: {} A:" # "{} The answer is"
genread_template = "Generate a background document from Wikipedia to answer the given question. {}"  # This prompt comes from the GenRead paper

def call_request(prompt, model, tokenizer, max_new_tokens=15):
    max_inpt_tokens = tokenizer.model_max_length
    if len(prompt) > tokenizer.model_max_length:  # conservative lower bound, since each token is at least 1 character
        inpts = tokenizer(prompt, return_tensors="pt")
        new_prompt = tokenizer.decode(inpts.input_ids[0, -(max_inpt_tokens - max_new_tokens):])
    else:
        new_prompt = prompt

    # try to get a response from the model multiple times if theres a timeout
    for i in range(5):
        try:
            if i > 0:
                print("Retrying request")
            response = openai.Completion.create(model=model, prompt=new_prompt, temperature=0.0, max_tokens=max_new_tokens, logprobs=5, top_p=1,frequency_penalty=0.0,presence_penalty=0.0)
            break
        except Exception as e:
            print(e)
            print("Timeout, trying again")
    
    pred = response["choices"][0]["text"]
    if pred.startswith("\n\n"):
        pred = pred[2:]
    pred = pred.split("\n")[0]
    return pred, response.to_dict_recursive()

def call_model(prompt, model, tokenizer, device, max_new_tokens=15, model_max_length=None):
    max_inpt_tokens = tokenizer.model_max_length if model_max_length is None else model_max_length
    inpts = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(input_ids=inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):], attention_mask=inpts.attention_mask[:, -(max_inpt_tokens - max_new_tokens):], pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    text = tokenizer.decode(gen[0])
    actual_prompt = tokenizer.decode(inpts.input_ids[0, -(max_inpt_tokens - max_new_tokens):])
    pred = text[len(actual_prompt):]
    if pred.startswith("\n\n"):
        pred = pred[2:]
    pred = pred.split("\n")[0]
    return pred, text

def clip_paragraph(text, eval_method):
    if eval_method in ["BM25", "genread"]:
        return text
    split = text.split(". ")
    return ". ".join(split[:-1]) + "."

def get_few_shot_text_with_retrieval(row, retrieval_dict, eval_method):
    if eval_method == "vanilla":
        return completion_template.format(row.question) + " " + row.obj
      # retrieval_dict[row.id]["ctxs"][0]
    if row.question.replace("?", "").lower() not in retrieval_dict:
        print("missing retrieval")
        return completion_template.format(row.question) + " " + row.obj
    else:
        retrieval = retrieval_dict[row.question.replace("?", "").lower()]["ctxs"][0]
        retrieved_text = clip_paragraph(retrieval["text"], eval_method)
        return retrieved_text + "\n\n" + completion_template.format(row.question) + " " + row.obj

def get_few_shot_text(row, eval_method):
    return completion_template.format(row.question) + " " + row.obj

def get_genread_passage(question, genread_template, generate_function, max_new_tokens=150):
    prompt = genread_template.format(question)
    return generate_function(prompt, max_new_tokens=max_new_tokens)[0]

def get_few_shot_examples_genread(knowledge, generate_function, n_examples, genread_template, is_templatedQA, max_new_tokens=150):
    if is_templatedQA:
        few_shot_examples = dict()
        all_pids = list(q_templates.keys())
        examples_per_template = n_examples // (len(q_templates) - 1)
        for pid in all_pids:
            for row2 in knowledge[knowledge.prop_id == pid].sample(n=examples_per_template).iloc:
                if pid not in few_shot_examples:
                    few_shot_examples[pid] = []
                generation = get_genread_passage(row2.question, genread_template, generate_function, max_new_tokens=max_new_tokens)
                few_shot_examples[pid].append(get_few_shot_text_with_retrieval(row2, {row2.question: {"ctxs": [{"id": -1, "text": generation}]}}, "genread"))
    else:
        few_shot_examples = []
        for row2 in knowledge.sample(n=n_examples + 1).iloc:
            generation = get_genread_passage(row2.question, genread_template, generate_function, max_new_tokens=max_new_tokens)
            few_shot_examples.append(get_few_shot_text_with_retrieval(row2, {row2.question: {"ctxs": [{"id": -1, "text": generation}]}}, "genread"))
    
    return few_shot_examples

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--alias', type=str)
    parser.add_argument('--n_examples', type=int, default=15)
    parser.add_argument('--eval_method', type=str, default="vanilla", choices=["vanilla", "BM25", "contriever", "genread"])
    parser.add_argument('--ret_path', type=str, default=None, required=False, help="path to retrieved documents jsonl")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--sample', type=int, default=0, help="if 0, use all examples")
    parser.add_argument('--continue_from', type=str, help="path to previous results file")
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument('--parallel', type=str, help="string of format 'i.n_workers' where i is the index of the worker")

    args = parser.parse_args()
        
    use_gpt3 = args.model_name in {"text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"}
    if use_gpt3:
        with open("../../openAIkey.txt") as f:
            openai.api_key = f.read()[:-1]
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        generate = lambda prompt, max_new_tokens: call_request(prompt, args.model_name, tokenizer, max_new_tokens=max_new_tokens)
    else:
        gpt = args.model_name
        device = args.device
        tokenizer = AutoTokenizer.from_pretrained(gpt)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if args.int8bit:
            model =  convert_model_to_int8_on_gpu(AutoModelForCausalLM.from_pretrained(gpt), device)
        else:
            model = AutoModelForCausalLM.from_pretrained(gpt).eval().to(device)
        if "opt" in args.model_name or args.model_name == "EleutherAI/gpt-neox-20b":
            generate = lambda prompt, max_new_tokens: call_model(prompt, model=model, tokenizer=tokenizer, device=device, max_new_tokens=max_new_tokens, model_max_length=2048)
        else:
            generate = lambda prompt, max_new_tokens: call_model(prompt, model=model, tokenizer=tokenizer, device=device, max_new_tokens=max_new_tokens)
    input_path = args.input_file
    knowledge = pd.read_csv(input_path, sep="\t")

    if args.continue_from is not None:
        results = pd.read_csv(args.continue_from, sep="\t")
        knowledge = knowledge[~knowledge.id.isin(results.id)]
    n = len(knowledge) if args.sample == 0 else args.sample
    sample = knowledge.sample(n=n, replace=False)
    if args.parallel is not None:
        worker_num, n_workers = map(int, args.parallel.split("."))
        sample = sample.iloc[worker_num::n_workers]

    n_examples = args.n_examples
    is_templatedQA = True
    examples_per_template = n_examples // (len(q_templates) - 1)

    preds = []
    prompts =[]
    accuracy = []
    responses = []
    if args.eval_method in ["BM25", "contriever"]:
        has_answer = []
        retrieval_ids = []
        with open(args.ret_path) as f:
            retrieval_dict = {json.loads(s)["question"]: json.loads(s) for s in f.readlines()}
        # print(retrieval_dict)
    if args.eval_method == "genread":
        genread_few_shot_examples = get_few_shot_examples_genread(knowledge, generate, n_examples, genread_template, is_templatedQA, max_new_tokens=150)
        has_answer = []
        gen_passages = []

    # main loop
    for row in tqdm(sample.iloc, total=n):

        # get few shot examples text
        if n_examples == 0:
            few_shot_examples_text = ""
        else:
            few_shot_examples = []
            if args.eval_method == "genread":
                if is_templatedQA:
                    other_pids = list(q_templates.keys())
                    other_pids.remove(row.prop_id)
                    few_shot_examples = []
                    for pid in other_pids:
                        few_shot_examples.extend(random.sample(genread_few_shot_examples[pid], examples_per_template))
                else:
                    few_shot_examples = random.sample([ex for ex in genread_few_shot_examples if row.question not in ex], n_examples)
            else:
                if is_templatedQA:
                    other_pids = list(q_templates.keys())
                    other_pids.remove(row.prop_id)
                    for pid in other_pids:
                        for row2 in knowledge[knowledge.prop_id == pid].sample(n=examples_per_template).iloc:
                            few_shot_examples.append(get_few_shot_text_with_retrieval(row2, retrieval_dict, args.eval_method) if args.eval_method in ["BM25", "contriever"] else get_few_shot_text(row2, args.eval_method))
                else:
                    for row2 in knowledge[knowledge.question != row.question].sample(n=n_examples).iloc:
                        few_shot_examples.append(get_few_shot_text_with_retrieval(row2, retrieval_dict, args.eval_method) if args.eval_method in ["BM25", "contriever"] else get_few_shot_text(row2, args.eval_method))
                
                    
            np.random.shuffle(few_shot_examples)
            few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"

        # get prompt
        if args.eval_method == "vanilla":
            prompt = few_shot_examples_text + completion_template.format(row.question)
        elif args.eval_method in ["BM25", "contriever"]:
            query = row.question
            try: 
                retrieval = retrieval_dict[query]["ctxs"][0]  # retrieval_dict[row.id]["ctxs"][0]
            except:

                print("No retrieval for", query, " Example query:", list(retrieval_dict.keys())[0])
                retrieval = {"text": "", "id": np.nan, "hasanswer": False}
            retrieved_text = clip_paragraph(retrieval["text"], eval_method=args.eval_method)
            retrieval_id = retrieval["id"]
            prompt = few_shot_examples_text + retrieved_text + "\n\n" + completion_template.format(row.question)
            has_answer.append(retrieval["hasanswer"])
            retrieval_ids.append(retrieval_id)
        elif args.eval_method == "genread":
            generation = get_genread_passage(row.question, genread_template, generate, max_new_tokens=150)
            prompt = few_shot_examples_text + generation + "\n\n" + completion_template.format(row.question)
            gen_passages.append(generation)
        
        # generate response
        pred, response = generate(prompt, max_new_tokens=args.max_new_tokens)
        prompts.append(prompt)
        preds.append(pred)
        responses.append(response)

        # compute accuracy
        possible_answers = json.loads(row.possible_answers)        
        is_correct = False
        genread_has_answer = False
        for pa in possible_answers:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
            if args.eval_method == "genread" and pa in response or pa.lower() in response or pa.capitalize() in response:
                genread_has_answer = True
        accuracy.append(is_correct)
        if args.eval_method == "genread":
            has_answer.append(genread_has_answer)

        # save results intermittently
        if len(preds) % 100 == 0:
            temp_sample = sample.iloc[:len(preds)].copy()
            temp_sample["pred"] = preds
            temp_sample["prompt"] = prompts
            temp_sample["generation"] = responses
            temp_sample["is_correct"] = accuracy
            if args.eval_method in ["BM25", "contriever"]:
                temp_sample["has_answer"] = has_answer
                temp_sample["retrieval_id"] = retrieval_ids
            if args.eval_method == "genread":
                temp_sample["has_answer"] = has_answer
                temp_sample["gen_passage"] = gen_passages
            model_name_alias = args.model_name.replace("/","_")
            if not os.path.exists(f"results/temp/"):
                os.makedirs(f"results/temp/")
            worker_str = "" if args.parallel is None else f"-worker={args.parallel}"
            output_path = f"results/temp/model={model_name_alias}-input={args.alias}-method={args.eval_method}-shots={n_examples}-n={len(temp_sample)}{'_int8bit' if args.int8bit is True else ''}{worker_str}.csv"
            temp_sample.to_csv(output_path, index=False)

    sample["is_correct"] = accuracy
    sample["prompt"] = prompts
    sample["pred"] = preds
    sample["generation"] = responses
    if args.eval_method in ["BM25", "contriever"]:
        sample["has_answer"] = has_answer
        sample["retrieval_id"] = retrieval_ids
    if args.eval_method == "genread":
        sample["has_answer"] = has_answer
        sample["gen_passage"] = gen_passages

    print(sample.is_correct.mean())
    model_name_alias = args.model_name.replace("/","_")
    worker_str = "" if args.parallel is None else f"-worker={args.parallel}"
    sample.to_csv(f"results/model={model_name_alias}-input={args.alias}-method={args.eval_method}-shots={n_examples}-n={len(sample)}{'_int8bit' if args.int8bit is True else ''}{worker_str}.csv")

        
if __name__ == "__main__":
    main()
