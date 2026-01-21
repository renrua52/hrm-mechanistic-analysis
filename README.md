# HRM: Are They Reasoning or Guessing?

Mechanistic analysis of **Hierarchical Reasoning Models (HRM)** on the *Sudoku-Extreme* dataset.  
This repository provides the **official PyTorch implementation** of our 2026 paper:

> **"Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models"**  
> *Zirui Ren, Ziming Liu*  
> arXiv:2601.10679 | [PDF](https://arxiv.org/abs/2601.10679)

## Dataset Preparation

~~~
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000-hint  --subsample-size 1000 --num-aug 1000 --hint 
python dataset/build_maze_dataset.py
~~~

## Training

~~~
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=40000 eval_interval=1000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000-hint epochs=40000 eval_interval=1000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
~~~

## Evaluation

Check W&B results.

## Reasoning Trace Analysis & Visualization

We added a `require_trace' argument to the HRM model forwarding, with the intermediate z_H states returned as a list.