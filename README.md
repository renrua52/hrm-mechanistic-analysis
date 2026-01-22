# HRM: Are They Reasoning or Guessing?

<img width="993" height="359" alt="image" src="https://github.com/user-attachments/assets/2425608d-f56f-4c7a-b426-f7e1227202dc" />

Mechanistic analysis of **Hierarchical Reasoning Models (HRM)** on the *Sudoku-Extreme* dataset.  
This repository provides the **official PyTorch implementation** of our 2026 paper:

> **"Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models"**  
> *Zirui Ren, Ziming Liu*  
> arXiv:2601.10679 | [PDF](https://arxiv.org/abs/2601.10679)

## Quick Start 🚀

Check out the [demo](./demo.ipynb) to understand how most results in the paper were attained.

## Dataset Preparation

Run the following commands to build vanilla and augmented sudoku dataset, respectively.

~~~
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000-hint  --subsample-size 1000 --num-aug 1000 --hint
~~~

The latter is an augmented version of the former, with easier sudoku puzzles mixed in. In the paper, we showed that this augmentation helps to restore inference stability.

## Training

Run the following commands to train HRM on either version of the datasets. Training randomness has observable impact on the outcome, so we recommend inspecting the `exact_accuracy` in W&B and choosing the best checkpoint for evaluation.

The `eval_interval` option does not influence training process. Evaluation typically takes considerable time, so set it wisely.

~~~
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=40000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000-hint epochs=40000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
~~~

## Model Evaluation

Most of the utilities for testing are implemented in `eval_utils.py`.



For evaluating trained checkpoints, we provide a quick-and-dirty implementation of batched inference in `batch_inference.py`, which runs on a single GPU. It supports the evaluation on *Sudoku-Extreme* of ensembled checkpoints, with or without the permuting token augmentation. For example, to do a full evaluation of the example checkpoint, run:
~~~
python batch_inference.py --checkpoints "checkpoint_example/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 pastel-lorikeet/example_checkpoint"
~~~
which typically takes 30 mins on an A10 GPU to reproduces the ~55% accuracy in the original HRM paper.

Multiple forwarding requires way more time. To get a taste of the accuracies of Table 1 in the paper, we recommend running the single-GPU script on a small portion of test samples. You can do something like:
~~~
python batch_inference.py --checkpoints "checkpoint_example/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 pastel-lorikeet/example_checkpoint" --permutes 9 --num_batch 10 --batch_size 100
~~~
which tests the designated ckpt on 1000 test samples, applying 9 token permutations to each.

For the full result of **Augmented HRM**, train your own series of checkpoints on the *augmented* dataset with ckpt interval 1000. Replace the checkpoints in `scripts/example_evaluation.sh` with your own. Then run the script. The full evaluation process takes around 500 GPU hours, due to the cost of multiple forward processes (90x). Again, if you just want to understand how it works, use a smaller number of samples.

Due to large variances in small-sample training, a ~2% discrepancy in single ckpt results and ~4% in multiple ckpt results are considered acceptable.

## Reasoning Trace Analysis & Visualization

We added a `require_trace` argument to the HRM model forwarding process, with the intermediate z_H states returned as a list. The `visualization` module is used to visualize both reasoning trajectory and error landscape. 
