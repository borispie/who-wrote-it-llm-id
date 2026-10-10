#!/bin/bash
# Reproduce everything (about 4-5 hours on a 2-core laptop CPU; much faster with more cores).
# rq1_rq2.py and rq3_cross_task.py skip any run whose result file already exists. To retrain from scratch,
# delete results/rq1_rq2/, results/rq3/ and results/rq3_parts/ first.
set -e
cd "$(dirname "$0")"
python src/build_dataset.py                                      # data/llm_responses.csv
python src/baselines.py                                          # chance + TF-IDF/logistic regression
python src/rq1_rq2.py --models cnn                               # RQ1 + RQ2: CNN, 3 input modes x 3 seeds
python src/rq1_rq2.py --models lstm --modes output --seeds 1 2 3  # RQ1: BiLSTM, 3 seeds
python src/rq1_rq2.py --models lstm --modes input_output --seeds 1     # RQ2: BiLSTM (1 seed: ~1 h on CPU)
python src/rq1_rq2.py --models lstm --modes input --seeds 1       # RQ2: BiLSTM prompt only
python src/rq2_biased.py                                         # RQ2: simulated biased data collection
python src/rq3_cross_task.py --models cnn --seeds 1 2 3          # RQ3: cross-task generalization (CNN)
python src/rq3_cross_task.py --models lstm --shifts writing --seeds 1   # RQ3 (BiLSTM)
python src/rq4_features.py                                       # RQ4: style features
python src/rq4_ablation.py --seeds 1 2 3                         # RQ4: remove one signal at a time
LLMID_THREADS=1 python src/rq4_cnn_filters.py                    # RQ4: windows the CNN's strongest filters respond to
python src/make_figures.py                                       # figures/ and report/tables/
