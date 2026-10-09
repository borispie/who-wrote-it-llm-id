"""Build the (LLM_name, LLM_input, LLM_output) dataset from public AlpacaEval outputs.

Source: tatsu-lab/alpaca_eval (Apache-2.0), results/<model>/model_outputs.json,
pinned to commit cd543a149df89434d8a54582c0151c0b945c3d20.
Every model answered the same 805 AlpacaEval instructions, so the classes are balanced.

Run:  python src/build_dataset.py            (downloads if data/raw_alpacaeval is missing)
Output: data/llm_responses.csv
"""
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

COMMIT = "cd543a149df89434d8a54582c0151c0b945c3d20"
URL = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/{commit}/results/{model}/model_outputs.json"

# one model per LLM family
MODELS = {
    "GPT": "gpt-4o-2024-05-13",
    "Claude": "claude-3-5-sonnet-20240620",
    "Gemini": "gemini-pro",
    "Llama": "Meta-Llama-3-70B-Instruct",
    "Qwen": "Qwen2-72B-Instruct",
    "DeepSeek": "deepseek-llm-67b-chat",
}

RAW_DIR = os.path.join("data", "raw_alpacaeval")
OUT_CSV = os.path.join("data", "llm_responses.csv")

# simple keyword rules for the task type of each prompt (used for RQ3)
CODING = re.compile(
    r"\b(code|coding|python|javascript|typescript|java|c\+\+|c#|sql|html|css|regex|bash|shell|script|"
    r"function|program|programming|algorithm|api|debug|compile|json|latex|react|node\.?js|golang|rust|"
    r"matlab|git|linux|excel formula|dataframe|pandas|numpy)\b",
    re.I,
)
MATH = re.compile(
    r"\b(math|calculate|compute|solve|equation|integral|derivative|probability|percent|percentage|"
    r"algebra|geometry|arithmetic|prime number|sum of|area of|volume of|how many|square root)\b"
    r"|\d+\s*[+*/×^=]\s*\d+|\d+\s+-\s+\d+",   # arithmetic like "3x + 10"; NOT ranges like "4-8"
    re.I,
)
WRITING = re.compile(
    r"\b(write|rewrite|compose|draft|poem|poetry|story|essay|letter|email|e-mail|lyrics|song|haiku|"
    r"limerick|slogan|tweet|blog|caption|joke|speech|dialogue|script|paraphrase|summarize|summary|"
    r"edit|proofread|rephrase|headline|title for|description for|ad copy|advertisement|creative)\b",
    re.I,
)


# Manual audit: I read every prompt the rules put in "coding" or "math" and fixed the false alarms below
# (e.g. "script" in "write a script for a YouTube video", "4-8" in an age range). Key = start of the prompt.
OVERRIDES = {
    # rules said coding
    "A job description is a document": "writing",
    "Can you explain to me how the stable diffusion algorithm": "qa_advice",
    "Create a detailed caption for an Instagram post": "writing",
    "Create a short, concise summary of the paper": "writing",
    "Curate a Spotify playlist": "qa_advice",
    "Extract the method that has been used in the research": "qa_advice",
    "Give the provided brand a motto": "writing",
    "I would like you to act as a virtual assistant": "qa_advice",
    "I'm currently studying Bioengineering": "qa_advice",
    "It would be helpful if you could suggest an acronym": "writing",
    "React properly to reviews from your customers": "writing",
    "Structure a podcast script": "writing",
    "Use an appropriate format to structure a formal letter": "writing",
    "We have entered the home supplies budget": "math",
    "What if Alan Turing had not cracked": "qa_advice",
    "Write a script for a YouTube video": "writing",
    "You are a script-writer": "writing",
    "You are given some reviews for a movie": "writing",
    "You will need to guide this person through the scenario": "qa_advice",
    "there used to be a program for winmx": "qa_advice",
    "what are the possible performance issues in a learning program": "qa_advice",
    "write a inspirational monologue script": "writing",
    "draw a man using ASCII characters": "writing",
    # rules said math
    "Hi, I'm trying to solve a crossword puzzle": "qa_advice",
    "How many black holes are known": "qa_advice",
    "How many days is it until Christmas": "qa_advice",
    "Predict how many stars the author will give": "qa_advice",
    "Provide a name for the dish given the ingredients": "writing",
    "Summarize the article you have been given in a brief manner": "writing",
    "Using critical thinking methods, how would you approach": "qa_advice",
    # missed by the rules (word problems with $ amounts, matrices, rewriting)
    "Given two matrices A and B": "math",
    "I bought two shirts from the store": "math",
    "Marie is at the music store": "math",
    "Marley has $20 left": "math",
    "Mick pays his teacher $800": "math",
    "Navina has $30 more": "math",
    "Using a given amount, determine an appropriate tip": "math",
    'Simplify "Most of the basic functions': "writing",
}


def task_type(instruction: str) -> str:
    """Coding first, then math, then writing; everything else is open QA / advice."""
    for prefix, label in OVERRIDES.items():
        if instruction.startswith(prefix):
            return label
    if CODING.search(instruction):
        return "coding"
    if MATH.search(instruction):
        return "math"
    if WRITING.search(instruction):
        return "writing"
    return "qa_advice"


def load_model(model: str):
    path = os.path.join(RAW_DIR, f"{model}.json")
    if not os.path.exists(path):
        os.makedirs(RAW_DIR, exist_ok=True)
        urllib.request.urlretrieve(URL.format(commit=COMMIT, model=model), path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(seed: int = 448):
    rows = []
    for family, model in MODELS.items():
        for rec in load_model(model):
            rows.append(
                {
                    "llm_family": family,
                    "llm_name": model,
                    "llm_input": rec["instruction"].strip(),
                    "llm_output": (rec["output"] or "").strip(),
                    "prompt_source": rec["dataset"],
                }
            )
    df = pd.DataFrame(rows)

    # one id per unique prompt (all 6 models answered the same prompts)
    prompts = sorted(df["llm_input"].unique())
    pid = {p: i for i, p in enumerate(prompts)}
    df["prompt_id"] = df["llm_input"].map(pid)

    # drop prompts where any model gave an empty answer, for ALL models (keeps classes balanced)
    bad = df.loc[df["llm_output"] == "", "prompt_id"].unique()
    df = df[~df["prompt_id"].isin(bad)].copy()
    print(f"dropped {len(bad)} prompt(s) with an empty output")

    df["task_type"] = df["llm_input"].map(task_type)

    # train / val / test split BY PROMPT (70/15/15), so a test prompt is never seen in training
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(df["prompt_id"].unique()))
    rng.shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    split = {i: "train" for i in ids[:n_tr]}
    split.update({i: "val" for i in ids[n_tr : n_tr + n_va]})
    split.update({i: "test" for i in ids[n_tr + n_va :]})
    df["split"] = df["prompt_id"].map(split)

    df = df.sort_values(["prompt_id", "llm_family"]).reset_index(drop=True)
    cols = ["prompt_id", "llm_family", "llm_name", "llm_input", "llm_output", "prompt_source", "task_type", "split"]
    df[cols].to_csv(OUT_CSV, index=False)

    print(f"saved {OUT_CSV}: {len(df)} rows, {df['prompt_id'].nunique()} prompts, {df['llm_family'].nunique()} families")
    print(df.groupby("split")["prompt_id"].nunique().rename("prompts").to_string())
    print(df.drop_duplicates("prompt_id")["task_type"].value_counts().rename("prompts per task type").to_string())
    print(df.drop_duplicates("prompt_id")["prompt_source"].value_counts().rename("prompts per source").to_string())


if __name__ == "__main__":
    main()
