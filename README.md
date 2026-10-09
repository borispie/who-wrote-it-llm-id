# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project. **Siyi Luo** (team leader and only member; individual project).

**Report:** [`Project_Report.pdf`](Project_Report.pdf)

Can a machine learning model tell which large language model (LLM) wrote a piece of text? This repository
trains a **CNN** and an **RNN (bidirectional LSTM)** in PyTorch to classify responses into six LLM families
(GPT, Claude, Gemini, Llama, Qwen, DeepSeek) and studies all four research questions:

| RQ | Question | Short answer |
|---|---|---|
| RQ1 | Which LLM generated a response? | **Yes, well above chance.** CNN 63.0%, BiLSTM 59.6%, TF-IDF baseline 63.3%, majority vote of all models 67.6% (chance 16.7%). Claude and Llama are easiest; Qwen is hardest (confused with GPT and DeepSeek). |
| RQ2 | Does the user's prompt help? | **No.** Prompt only = chance (16.7%), because every LLM answered the same prompts; prompt + response ≈ response only. A simulated *biased* collection lets prompts alone reach ~60%, so prompt-only accuracy is a good test for dataset bias. |
| RQ3 (extra) | Do fingerprints carry over to unseen task types? | **Mostly.** Never seeing writing (or coding) costs only 2–3 points vs. a size-matched control, but writing and coding answers are harder to attribute in general (~47–51% vs ~66%). Claude and Llama stay recognizable; GPT loses its markdown clues on writing. |
| RQ4 (extra) | What distinguishes the LLMs? | **Mostly structure.** With every word replaced by a placeholder, a CNN still gets 61.3%; with words only it gets 46.6%. Habits: GPT uses headers, bold and curly apostrophes; Claude opens with "Here's" and almost never uses bold; Llama opens with "What a great question!" and uses `*` bullets. |

## Data

`data/llm_responses.csv`: 4,824 rows of `(llm_family, llm_name, llm_input, llm_output)` plus `prompt_id`,
`prompt_source`, `task_type` and `split`.

* **Source:** public model outputs from [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) (Apache-2.0),
  `results/<model>/model_outputs.json`, pinned to commit `cd543a1`. No private data; I did not collect any
  conversations myself.
* **Models (one per family):** `gpt-4o-2024-05-13`, `claude-3-5-sonnet-20240620`, `gemini-pro`,
  `Meta-Llama-3-70B-Instruct`, `Qwen2-72B-Instruct`, `deepseek-llm-67b-chat`.
* **Balanced by design:** every model answered the same 804 prompts (one prompt with an empty answer was dropped
  for all models), so there are 804 responses per family.
* **Task types** (for RQ3): keyword rules plus a manual check of every coding/math prompt (38 labels fixed by
  hand, listed in `src/build_dataset.py`): 549 QA/advice, 162 writing, 60 coding, 33 math.
* **Split by prompt** (70/15/15): all six answers to a prompt are in the same split.

## How to run

```bash
pip install -r requirements.txt
bash run_all.sh          # or run the steps below one by one
```

| Step | Command | Output |
|---|---|---|
| Build dataset | `python src/build_dataset.py` | `data/llm_responses.csv` |
| Baselines | `python src/baselines.py` | `results/baselines.json` |
| RQ1 + RQ2 | `python src/rq1_rq2.py --models cnn` and the BiLSTM lines in `run_all.sh` | `results/rq1_rq2/`, `results/rq1_rq2_summary.csv` |
| RQ2 bias test | `python src/rq2_biased.py` | `results/rq2_biased.json` |
| RQ3 | `python src/rq3_cross_task.py --models cnn --seeds 1 2 3` and `--models lstm --shifts writing --seeds 1` | `results/rq3/` |
| RQ4 features | `python src/rq4_features.py` | `results/rq4_*.csv`, `results/rq4_features.json` |
| RQ4 ablations | `python src/rq4_ablation.py` | `results/rq4_ablation.json` |
| RQ4 CNN filters | `LLMID_THREADS=1 python src/rq4_cnn_filters.py` | `results/rq4_cnn_filters.json` |
| Figures + tables | `python src/make_figures.py` | `figures/`, `report/tables/` |

Everything runs on a CPU. The full pipeline took about 4–5 hours on a 2-core machine with no GPU (one BiLSTM run takes
about 40 minutes; one CNN run about 5). Set `LLMID_THREADS` to control the number of PyTorch threads. The CNN uses 3
seeds everywhere; because of its cost, the BiLSTM uses 3 seeds for RQ1 and 1 seed for the other experiments.

## Repository layout

```
src/
  build_dataset.py   download AlpacaEval outputs, label task types, split by prompt
  llmid.py           tokenizer, vocabulary, CNN, BiLSTM, training loop, metrics (shared by all scripts)
  baselines.py       chance + TF-IDF / logistic regression
  rq1_rq2.py         CNN + BiLSTM on output only / input only / input + output
  rq2_biased.py      simulated biased data collection (when does the prompt leak the label?)
  rq3_cross_task.py  train on some task types, test on an unseen one (with a size-matched control);
                     --part/--merge lets the three trainings of one run go in parallel
  rq4_features.py    style statistics + style-feature classifiers
  rq4_ablation.py    remove one kind of signal at a time (test-time and retrained)
  rq4_cnn_filters.py which text windows the CNN's strongest filters respond to
  make_figures.py    all figures and tables (the report uses a subset of them)
data/                the dataset (CSV)
results/             all metrics, predictions and summaries (JSON/CSV)
figures/             all figures (the report uses two of them)
logs/                training logs (loss and validation scores per epoch; queue_lstm.log also
                     holds the partial lines of BiLSTM runs I stopped and restarted to reorder the queue)
report/              LaTeX source of the report
Project_Report.pdf   the report
```

## Models

* **CNN** (Kim, 2014): 128-d embeddings, 1-D convolutions of width 3/4/5 with 100 filters each, masked max-pooling
  over time, dropout 0.5, linear layer.
* **BiLSTM:** 128-d embeddings, 1-layer bidirectional LSTM (64 units per direction) with packed sequences,
  mean + max pooling, dropout 0.5, linear layer.
* **Input:** a regex tokenizer that keeps line breaks and markdown (`**`, `#`, ```` ``` ````) as tokens; first 200 +
  last 50 tokens of each response. Vocabulary built from the training split only.
* **Training:** AdamW (lr 1e-3, weight decay 1e-4), batch 32, cross-entropy, gradient clipping 1.0, up to 15 epochs,
  early stopping on validation macro-F1 (patience 3), seeds 1–3.
