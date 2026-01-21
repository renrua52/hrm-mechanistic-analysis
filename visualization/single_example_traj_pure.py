#!/usr/bin/env python3
"""
Script to load an HRM checkpoint and forward a single hand-coded example
to get output logits and predictions.
"""

import argparse
import yaml
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

import torch
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Import necessary components from existing files
from pretrain import PretrainConfig, init_train_state, create_dataloader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch

def plot_tensor_trajectory(tensor_list, idx, arrow_scale=100.0, linewidth=2.5,
                           colormap='plasma', figsize=(10, 8)):
    """
    Minimal version - only arrows, no points or labels
    """
    
    # Validate input
    if not isinstance(tensor_list, list) or len(tensor_list) < 2:
        raise ValueError("Input must be a list with at least 2 tensors")
    
    n_tensors = len(tensor_list)
    
    # Flatten tensors
    flattened_vectors = [tensor.reshape(1, -1) for tensor in tensor_list]
    data_matrix = np.vstack(flattened_vectors)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_matrix)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_coords = pca_result[:, 0]
    y_coords = pca_result[:, 1]
    
    # Color gradient
    cmap = plt.cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, n_tensors - 1))
    
    # Draw arrows only
    for i in range(n_tensors - 1):
        start_x, start_y = x_coords[i], y_coords[i]
        end_x, end_y = x_coords[i + 1], y_coords[i + 1]
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        ax.arrow(
            start_x, start_y, 
            dx, dy,
            head_width=0.03 * arrow_scale,
            head_length=0.05 * arrow_scale,
            fc=colors[i],
            ec=colors[i],
            linewidth=linewidth,
            length_includes_head=True,
            alpha=0.8,
            shape='full'
        )
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    mid_x = (xlim[0] + xlim[1]) / 2
    mid_y = (ylim[0] + ylim[1]) / 2
    half  = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    
    # Simple labels
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Reasoning Trajectory')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)
    
    # ---------- 新增 colorbar ----------
    # 1. 生成一个 ScalarMappable 对象，供 colorbar 使用
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=1, vmax=n_tensors-1))
    sm.set_array([])          # 仅用于映射，无需真实数据
    
    # 2. 在 figure 右侧添加 colorbar
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    
    # 标签
    cbar.set_label('Reasoning Step', fontsize=18)          # 18 pt，约 ICML \Large
    # 刻度值
    cbar.ax.tick_params(labelsize=16)            # 16 pt
    cbar.set_ticks(np.linspace(1, n_tensors-1, min(5, n_tensors-1)))
    cbar.set_ticklabels([f'{int(t)}' for t in cbar.get_ticks()])
    # ------------------------------------
    
    plt.tight_layout()
    plt.savefig(f"single_results/traj_pure/traj_{idx}.pdf", transparent=True)

def plot_tensor_trajectories(tensor_lists, idx, arrow_scale=100.0, linewidth=2.5, colormap='tab10', figsize=(10, 8)):
    """
    简化版本 - 只绘制轨迹线，无标记
    """
    
    if not isinstance(tensor_lists, list):
        raise ValueError("Input must be a list of tensor lists")
    
    n_trajectories = len(tensor_lists)
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用离散的颜色映射
    if colormap == 'tab10' and n_trajectories > 10:
        colormap = 'tab20'
    
    trajectory_colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_trajectories))
    
    for traj_idx, tensor_list in enumerate(tensor_lists):
        if not isinstance(tensor_list, list) or len(tensor_list) < 2:
            continue
        
        # 展平和PCA
        flattened = [t.reshape(1, -1) for t in tensor_list]
        data_matrix = np.vstack(flattened)
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(data_matrix)
        
        # 为整个轨迹使用单一颜色
        traj_color = trajectory_colors[traj_idx]
        
        # 绘制所有箭头
        for i in range(len(coords) - 1):
            start_x, start_y = coords[i]
            end_x, end_y = coords[i + 1]
            
            ax.arrow(
                start_x, start_y,
                end_x - start_x, end_y - start_y,
                head_width=0.03 * arrow_scale,
                head_length=0.05 * arrow_scale,
                fc=traj_color,
                ec=traj_color,
                linewidth=linewidth,
                length_includes_head=True,
                alpha=0.7
            )
    
    ax.set_aspect('equal')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Tensor Trajectories')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"single_results/traj_pure/traj_{idx}.pdf")

verbose = True
strict = True
import random
tests = range(100)
# tests=[8, 62, 82]

def load_checkpoint_and_config(checkpoint_path: str):
    """Load checkpoint and config, adapted from eval_checkpoint.py"""
    # Find the checkpoint directory
    if os.path.isfile(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_file = checkpoint_path
    else:
        # Assume it's a directory, find the latest checkpoint
        checkpoint_dir = checkpoint_path
        import glob
        pattern = os.path.join(checkpoint_dir, "step_*")
        checkpoint_files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Sort by step number and take the latest
        def extract_step(filepath):
            filename = os.path.basename(filepath)
            if filename.startswith("step_"):
                try:
                    return int(filename.removeprefix("step_"))
                except ValueError:
                    return 0
            return 0
        
        checkpoint_files.sort(key=extract_step)
        checkpoint_file = checkpoint_files[-1]
        print(f"Using latest checkpoint: {os.path.basename(checkpoint_file)}")
    
    # Load config from checkpoint directory
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    
    print(f"Loaded config from: {config_path}")
    print(f"Model: {config.arch.name}")
    print(f"Data path: {config.data_path}")
    
    return checkpoint_file, config, checkpoint_dir


def create_hand_coded_example(vocab_size: int, seq_len: int, puzzle_id: int = 0, idx = None, simplify = False):
    """Create a hand-coded example sequence for testing"""
    
    # Create input sequence
    if idx is not None:
        all_inputs = np.load("/cephfs/renzirui/projects/HRM-ds/data/sudoku-extreme-1k-aug-1000/test/all__inputs.npy")
        all_labels = np.load("/cephfs/renzirui/projects/HRM-ds/data/sudoku-extreme-1k-aug-1000/test/all__labels.npy")
        inputs = all_inputs[idx]
        labels = all_labels[idx]
        if simplify:
            inputs = np.array([x for x in labels])
            inputs[0] = 1
            for i in range(9):
                inputs[i] = 1
    else:
        inputs = np.array([1 for _ in range(81)])
        labels = np.array([0 for _ in range(81)])

    
    # Convert to tensors
    inputs_tensor = torch.from_numpy(inputs).long().unsqueeze(0)
    labels_tensor = torch.from_numpy(labels).long().unsqueeze(0)
    puzzle_ids_tensor = torch.from_numpy(np.asarray(puzzle_id)).long().unsqueeze(0)
    
    batch = {
        "inputs": inputs_tensor,
        "labels": labels_tensor,
        "puzzle_identifiers": puzzle_ids_tensor
    }
    
    if verbose:
        print(f"Created hand-coded example:")
        print(f"  Input sequence:\n {np.array(inputs).reshape(9,9)}")
        print(f"  Label sequence:\n {np.array(labels).reshape(9,9)}")
    
    return batch

def forward_z(model, z_H): #z_H: (82, 512)
    logits = model.model.inner.lm_head(z_H.to("cuda"))[1:] # (81, 11)
    return logits

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Carry, HierarchicalReasoningModel_ACTV1InnerCarry
def forward_single_example(model, batch: Dict[str, torch.Tensor], noise, max_steps: int = 1000):
    model.eval()
    
    # Move batch to device
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Initialize carry state (adapted from pretrain.py)
    with torch.device("cuda"):
        carry = model.initial_carry(batch)
    
    # Store results from each step
    all_logits = []
    all_predictions = []
    losses = []
    z_H_traces = []
    z_H_list = []
    
    step = 0
    
    with torch.inference_mode():
        while step < max_steps:
            with torch.no_grad():
                z_H_trace, carry, loss, metrics, preds, all_finish = model(
                    carry=carry, 
                    batch=batch, 
                    require_trace=True,
                    return_keys=["logits"]  # Request logits in return
                )
                z_H_traces.extend(z_H_trace)
                losses.append(loss)
            
            logits = preds["logits"]  # [1, seq_len, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # [1, seq_len]
            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            
            step += 1

            if all_finish:
                break

    
    return {
        "all_losses": losses,
        "all_logits": all_logits,
        "all_predictions": all_predictions,
        # "all_carries": all_carries,
        "total_steps": step,
        "z_H_list": z_H_list,
        "z_H_traces": z_H_traces,
    }

def print_results(batch: Dict[str, torch.Tensor], results: Dict[str, Any], trace=False):
    """Print detailed results from the forward pass"""
    # print(f"\n{'='*60}")
    # print("INFERENCE RESULTS")
    # print(f"{'='*60}")
    
    # Input information
    inputs = batch["inputs"].cpu().numpy()[0]
    labels = batch["labels"].cpu().numpy()[0]

    
    # print(f"Input sequence:\n{inputs.reshape(9,9)}\n")
    # print(f"Target labels:\n{labels.reshape(9,9)}\n")
    # print(f"Total reasoning steps: {results['total_steps']}")
    
    # Show predictions from each step
    # print(f"\nPredictions by step:")
    assert len(results["all_logits"]) == len(results["all_predictions"])

    flag = 0
    cnt = 0
    restore = -1

    success_steps = -1
    for step, preds in enumerate(results["all_predictions"]):
        pred_seq = preds.numpy()[0]
        logits = results["all_logits"][step]

        temperature = 1
        probs = torch.softmax(logits/temperature, dim=-1, dtype=torch.float64).squeeze(dim=0)

        confidence = np.array([probs[i, pred_seq[i]] for i in range(len(pred_seq))])

        # logits = torch.tensor(logits, dtype=torch.float32)
        # confidence = np.array([logits[0, i, pred_seq[i]] for i in range(len(pred_seq))])

        if not (False in (pred_seq == labels)):
            cnt += 1
            if not strict or step == 16 - 1:
                flag = 1
            
            if success_steps == -1:
                success_steps = step+1
        
            if cnt == 2:
                print(f"Restored after {step-success_steps+1} steps.")
                restore = step-success_steps+1
        print(f"Step {step+1:2d}: Incorrect" if False in (pred_seq == labels) else f"Step {step+1:2d}: Correct")
        if verbose:
            print(f"Step {step+1:2d}:\n{pred_seq.reshape(9, 9)}")

    if cnt < 2:
        print("Failed.")

    z_H_traces = results["z_H_traces"]
    if trace:
        all_zH_list = [z.squeeze(0).to("cuda") for z in z_H_traces]
    else:
        all_zH_list = None
    return flag, restore, all_zH_list


def main():
    parser = argparse.ArgumentParser(
        description="Load HRM checkpoint and forward a single hand-coded example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", required=True)
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Using CUDA device 0")
    else:
        print("CUDA not available, using CPU")
    
    try:
        # Load checkpoint and config
        checkpoint_file, config, checkpoint_dir = load_checkpoint_and_config(args.checkpoint_path)
        
        # Seed RNGs for consistency (same as pretrain.py)
        torch.random.manual_seed(config.seed)
        
        # Create a dummy dataloader to get metadata (adapted from eval_checkpoint.py)
        # We need this to get vocab_size, seq_len, etc.
        train_loader, train_metadata = create_dataloader(
            config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=1,  # Small batch for metadata only
            rank=0,
            world_size=1
        )
        
        # Initialize model (adapted from eval_checkpoint.py)
        train_state = init_train_state(config, train_metadata, world_size=1)
        
        # Load checkpoint weights (adapted from eval_checkpoint.py)
        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cuda")
        try:
            train_state.model.load_state_dict(checkpoint, assign=True)
        except:
            # Handle torch.compile wrapped models
            cleaned_state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}
            train_state.model.load_state_dict(cleaned_state_dict, assign=True)
        
        # Set model to evaluation mode
        train_state.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in train_state.model.parameters()):,}")
        
        
        correct = 0
        trajs = []
        # ---------- 1. ICML 字体/尺寸 ----------
        rc('font', family='serif', size=9)        # 9 pt 正文
        # rc('text', usetex=True)                   # 矢量字体
        plt.figure(figsize=(3.25, 2.2))           # 单栏宽，高宽比≈0.7

        # ---------- 2. 高对比配色 ----------
        green = '#00A652'     # 深荧光绿，打印不糊
        gray  = '#636363'     # 比 50% 黑再深一点
        loss_list = []
        for idx in tests:
            print(f'=====Testing sample {idx}=====')

            # Create hand-coded example
            batch = create_hand_coded_example(
                vocab_size=train_metadata.vocab_size,
                seq_len=train_metadata.seq_len,
                puzzle_id=0,
                idx=idx,
                simplify=True
            )
            inputs = batch["inputs"].squeeze(0)
            labels = batch["labels"].squeeze(0)

            results = forward_single_example(
                model=train_state.model,
                batch=batch,
                max_steps=10000,
                noise=0
            )

            flag, _, all_zH_list = print_results(batch, results, trace=True)
            trajs.append([z.cpu().float() for z in all_zH_list])
            losses = [torch.nn.CrossEntropyLoss()(forward_z(train_state.model, z), labels.to("cuda")).item() for z in all_zH_list]
            x_half = np.arange(len(losses)) / 2          # 实际 x 坐标的一半
            if flag:
                plt.plot(x_half, losses, color=green, lw=0.5, alpha=0.40)
            else:
                plt.plot(x_half, losses, color=gray,  lw=0.25, alpha=0.25)
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            loss_list.append(losses)

            plot_tensor_trajectory(trajs[-1], idx)
        # ---------- 4. 图例（短横，无框） ----------
        green_bar = plt.Line2D([0, 1], [0, 0], color=green, lw=3)
        gray_bar  = plt.Line2D([0, 1], [0, 0], color=gray,  lw=3)

        # ---------- 5. 轴样式 ----------
        plt.xlabel('Segment index', fontsize=9)
        plt.ylabel('Segment loss',  fontsize=9)
        plt.tick_params(axis='both', which='major', labelsize=8, direction='in')
        sns.despine(trim=True)            # 去掉右侧/顶部脊柱
        plt.grid(False)                   # ICML 不喜欢网格

        # ---------- 6. 保存 ----------
        plt.tight_layout(pad=0.2)
        plt.savefig('loss.pdf', format='pdf', transparent=True)  # 矢量，供LaTeX插入
        plt.savefig('loss.png', dpi=1200, bbox_inches='tight')   # 位图，供预览
        plt.close()



        losseses = np.array(loss_list).mean(axis=0)
        plt.figure(1, (8, 6))
        plt.plot(losseses)
        plt.savefig("loss_avg.png")
        plt.close()

        plot_tensor_trajectories(trajs, -1)


        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

'''
python single_example_noise.py --checkpoint_path "/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000-hint ACT-torch/HierarchicalReasoningModel_ACTV1 wonderful-trogon/step_48174" > single_inference_result.txt

python single_example_traj_pure.py --checkpoint_path "/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 pastel-lorikeet/step_26040" > single_inference_result.txt

python single_example_traj_pure.py --checkpoint_path "/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000-hint-rand-all ACT-torch/HierarchicalReasoningModel_ACTV1 burrowing-pudu/step_52080" > single_inference_result.txt

python single_example_noise.py --checkpoint_path "/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 optimal-mongrel" > single_inference_result.txt
'''
