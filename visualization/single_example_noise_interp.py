#!/usr/bin/env python3
"""
Script to load an HRM checkpoint and forward a single hand-coded example
to get output logits and predictions.
"""

import argparse
import yaml
import os
from typing import Dict, Any

import torch
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Import necessary components from existing files
from pretrain import PretrainConfig, init_train_state, create_dataloader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

def plot_ij_dict(data_dict, z_proj_list, idx, dests=None):
    """
    Generates a publication-quality 2D heatmap for ICML.
    - Uses LaTeX for text rendering and serif fonts.
    - Employs distinct markers for special points.
    - Uses hatching for invalid regions.
    """
    if not data_dict:
        raise ValueError("data_dict is empty, cannot plot.")

    # === 1. PUBLICATION-QUALITY STYLE CONFIGURATION ===
    # This should ideally be set once at the beginning of your script.
    # NOTE: text.usetex=True requires a working LaTeX installation.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        # "text.usetex": True, # For professional math-style fonts
        "pdf.fonttype": 42,   # Ensures fonts are embedded in the PDF
        "ps.fonttype": 42
    })

    # === 2. DATA PREPARATION (Unchanged) ===
    all_i, all_j = zip(*data_dict.keys())
    min_i, max_i = min(all_i), max(all_i)
    min_j, max_j = min(all_j), max(all_j)
    grid = np.full((max_j - min_j + 1, max_i - min_i + 1), np.nan, dtype=float)
    for (i, j), v in data_dict.items():
        grid[j - min_j, i - min_i] = float(v)

    # === 3. PLOTTING SETUP ===
    fig, ax = plt.subplots(figsize=(8, 7)) # Adjusted for better aspect ratio
    extent = (min_i - 0.5, max_i + 0.5, min_j - 0.5, max_j + 0.5)

    # === 4. DRAW PLOT LAYERS ===
    # Background for NaN values
    ax.imshow(np.ones_like(grid), cmap=ListedColormap(['#eeeeee']),
              origin='lower', extent=extent, zorder=0)

    # Main heatmap
    main_grid = np.ma.masked_where((grid == -1) | np.isnan(grid), grid)
    im = ax.imshow(main_grid, cmap='plasma', origin='lower', extent=extent, zorder=1)
    
    # Hatched mask for invalid (-1) regions - more professional than solid red
    hatch_mask = np.ma.masked_where(grid != -1, np.ones_like(grid))
    ax.pcolor(np.linspace(extent[0], extent[1], grid.shape[1] + 1), 
              np.linspace(extent[2], extent[3], grid.shape[0] + 1),
              hatch_mask, 
              hatch='//', alpha=0.0, zorder=2) # alpha=0 makes background transparent

    # === 5. OVERLAY ELEMENTS (Arrows and Paths, Unchanged) ===
    if z_proj_list:
        z_arr = np.asarray(z_proj_list)
        ax.plot(z_arr[:, 0], z_arr[:, 1], color='gray', lw=2, marker='o', 
                markersize=4, label='Optimization Path', zorder=8)
    if dests: # Your arrow plotting logic remains here
        pass # Add your quiver logic back if needed

    # === 6. MARK AND CONNECT SPECIAL POINTS ===
    special_points = {
        'True FP': {'coords': (-3, -1), 'color': '#00FFFF', 'marker': '*'}, # Cyan -> Star
        'Spurious FP': {'coords': (21, 1), 'color': '#FF6347', 'marker': 'X'}  # Tomato -> X
    }
    
    # Plot connecting line first (lower zorder)
    coords_a = special_points['True FP']['coords']
    coords_b = special_points['Spurious FP']['coords']
    ax.plot([coords_a[0], coords_b[0]], [coords_a[1], coords_b[1]],
            color='white', linestyle='--', linewidth=2.5, zorder=9)

    # Plot points on top of the line
    for label, props in special_points.items():
        ax.scatter(*props['coords'],
                   c=props['color'],
                   s=350,  # Increased size
                   marker=props['marker'],
                   edgecolors='black',
                   linewidths=1.5,
                   label=label,
                   zorder=10)

    # === 7. FINAL TOUCHES ===
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # ax.set_title('Error Landscape on the Principal Plane')
    ax.set_aspect('equal')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error Value')
    cbar.outline.set_edgecolor('gray')

    # Legend
    # Legend
    leg = ax.legend(loc='lower right', frameon=True)
    leg.get_frame().set_facecolor('white') # 设置背景为白色
    leg.get_frame().set_alpha(0.8)         # 设置背景半透明，可根据需要调整 (0.0 到 1.0)
    leg.get_frame().set_edgecolor('gray')  # 设置一个浅色边框
    
    plt.tight_layout(pad=0.5)
    plt.savefig("icml_figure.pdf", bbox_inches='tight')
    plt.close(fig)

# Example usage:
# plot_ij_dict_for_publication(my_data_dict, my_z_proj_list, 0)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

def plot_ij_dict_3d(data_dict, feiwu, idx, feiwuu):
    """
    将 data_dict 的数据绘制成三维地形图，并高亮标记两个特殊点。
    """
    if not data_dict:
        raise ValueError("data_dict is empty, cannot plot.")

    # 1. 提取坐标和数值
    keys = np.array(list(data_dict.keys()))
    values = np.array(list(data_dict.values()))
    
    all_i, all_j = keys[:, 0], keys[:, 1]
    min_i, max_i = int(min(all_i)), int(max(all_i))
    min_j, max_j = int(min(all_j)), int(max(all_j))

    # 2. 创建用于曲面图的坐标网格
    i_range = np.arange(min_i, max_i + 1)
    j_range = np.arange(min_j, max_j + 1)
    I, J = np.meshgrid(i_range, j_range)

    # 3. 构造 Z 轴网格，缺失值将由 griddata 自动处理（默认填充 nan）
    grid_z = griddata(keys, values, (I, J), method='linear')
    grid_z_main = np.where(grid_z == -1, np.nan, grid_z)
    
    # 4. 设置 3D 绘图环境
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 5. 绘制主曲面
    vmin = np.nanmin(grid_z_main)
    vmax = np.nanmax(grid_z_main)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    surf = ax.plot_surface(I, J, grid_z_main, cmap='plasma',
                           norm=norm, rstride=1, cstride=1, 
                           antialiased=True, shade=True, zorder=1, alpha=0.9)

    # 添加颜色条
    mappable = ScalarMappable(cmap='plasma', norm=norm)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cb.set_label('Error Value')

    # 6. 在底面标记值为 -1 的点
    z_min_plane = vmin if not np.isnan(vmin) else 0
    mask_indices = np.where(grid_z == -1)
    if mask_indices[0].size > 0:
        ax.scatter(I[mask_indices], J[mask_indices], z_min_plane,
                   c='red', marker='x', s=20, label='Value = -1', zorder=2)


    # 7. 绘制您指定的特定三维点
    # 定义要绘制的点，格式为 (i, j, z)
    # points_to_plot = {
    #     'Point A': {'coords': (-3, -1, 10), 'color': 'yellow'},
    #     'Point B': {'coords': (21, 1, 23), 'color': 'lime'}
    # }

    # for label, props in points_to_plot.items():
    #     # 从字典中解包坐标
    #     px, py, pz = props['coords']
        
    #     # 使用 ax.scatter 绘制点
    #     ax.scatter(px, py, pz, 
    #             c=props['color'], 
    #             s=150,               # 较大的尺寸以突出显示
    #             marker='o', 
    #             edgecolors='black',  # 添加黑色边框以增加可见性
    #             depthshade=False,    # 确保颜色不受光照影响
    #             label=label,         # 添加图例标签
    #             zorder=10)           # 确保点在曲面上方渲染

# ... 函数的其余部分（设置图例、标签等）...


    # 8. 设置图例、标签和布局
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Error Value')
    ax.set_title('Error Value Landscape on Principle Plane')
    ax.view_init(elev=30., azim=-120) # 可调整视角

    # ===== 9. 核心修正：手动设置坐标轴范围以确保所有点可见 =====
    # 获取曲面和特殊点的Z轴范围
    # z_points = [p['coords'][2] for p in points_to_plot.values()]
    z_min_overall = min(np.nanmin(grid_z_main), 0)
    z_max_overall = max(np.nanmax(grid_z_main), 50)
    
    # 设置包含所有元素的坐标轴范围，并增加一点边距
    ax.set_xlim(min_i, max_i)
    ax.set_ylim(min_j, max_j)
    ax.set_zlim(z_min_overall - 1, z_max_overall + 1)
    
    plt.tight_layout()
    plt.savefig(f"z_interp_3d_highlighted_{idx}.pdf", transparent=True)
    plt.close()





verbose = True
strict = True
correct_list = [(0, 3), (1, 5), (8, 10), (9, 4), (10, 3), (11, 7), (12, 1), (13, 14), (14, 8), (15, 5), (16, 4), (18, 15), (19, 3), (23, 6), (24, 5), (25, 11), (26, 4), (27, 3), (30, 2), (31, 2), (32, 5), (34, 3), (36, 16), (39, 3), (41, 3), (44, 5), (47, 5), (48, 4), (49, 5), (51, 4), (52, 7), (56, 5), (60, 2), (61, 5), (62, 4), (63, 3), (64, 2), (66, 4), (67, 2), (68, 3), (71, 3), (72, 7), (77, 14), (78, 3), (83, 5), (84, 5), (85, 8), (86, 12), (87, 5), (89, 3), (90, 3), (93, 2), (99, 4), (100, 4), (101, 13), (102, 4), (103, 3), (106, 3), (108, 3), (109, 3), (113, 5), (115, 3), (116, 5), (120, 4), (121, 3), (122, 4), (124, 9), (126, 6), (127, 3), (129, 3), (133, 7), (134, 9), (135, 9), (136, 5), (137, 7), (139, 3), (146, 6), (148, 4), (149, 4), (150, 7), (151, 4), (152, 12), (156, 4), (158, 4), (159, 7), (160, 11), (162, 7), (163, 6), (166, 5), (167, 5), (171, 3), (172, 5), (176, 2), (177, 9), (178, 4), (179, 3), (180, 3), (182, 12), (183, 2), (185, 2), (186, 3), (189, 11), (191, 3), (192, 8), (193, 3), (194, 4), (195, 16), (196, 4), (201, 3), (202, 2), (203, 4), (204, 12), (207, 5), (208, 7), (211, 1), (212, 6), (214, 3), (216, 5), (217, 3), (218, 4), (219, 2), (223, 3), (225, 3), (228, 8), (229, 4), (231, 5), (233, 3), (234, 3), (236, 5), (238, 3), (239, 15), (241, 4), (244, 6), (246, 6), (247, 3), (248, 4), (249, 4), (255, 10), (257, 3), (260, 2), (262, 2), (263, 4), (264, 4), (265, 8), (267, 4), (270, 1), (277, 4), (278, 7), (279, 5), (280, 3), (285, 12), (293, 5), (294, 8), (296, 3), (298, 3), (299, 8), (301, 7), (302, 4), (303, 4), (306, 9), (313, 11), (317, 3), (318, 3), (319, 2), (322, 3), (323, 3), (325, 4), (327, 9), (329, 4), (330, 8), (332, 11), (337, 12), (338, 2), (339, 5), (340, 7), (341, 5), (343, 3), (345, 3), (346, 5), (349, 3), (350, 15), (354, 5), (355, 5), (356, 4), (360, 5), (365, 4), (366, 3), (368, 5), (370, 7), (371, 5), (375, 5), (376, 9), (382, 5), (383, 4), (387, 4), (388, 3), (389, 7), (390, 6), (391, 9), (393, 2), (395, 3), (396, 3), (400, 2), (402, 5), (405, 2), (406, 10), (412, 6), (416, 3), (420, 13), (423, 6), (424, 9), (426, 9), (430, 4), (431, 3), (433, 3), (434, 2), (438, 3), (439, 6), (440, 12), (443, 4), (444, 8), (445, 4), (448, 2), (449, 5), (450, 1), (451, 11), (455, 6), (456, 2), (459, 9), (461, 4), (462, 6), (463, 7), (464, 2), (467, 4), (468, 4), (470, 5), (472, 5), (473, 7), (474, 9), (475, 1), (476, 3), (479, 2), (482, 11), (483, 5), (531, 14), (542, 16), (582, 2), (585, 1), (586, 4), (588, 5), (589, 4), (590, 8), (591, 4), (592, 4), (593, 3), (594, 3), (595, 9), (596, 2), (598, 4), (600, 7), (602, 4), (604, 9), (621, 5), (636, 9), (651, 7), (654, 8), (657, 5), (658, 3), (659, 2), (660, 13), (661, 2), (666, 5), (669, 4), (670, 8), (675, 3), (676, 3), (677, 2), (682, 3), (684, 1), (685, 6), (686, 5), (690, 3), (692, 3), (693, 6), (694, 3), (695, 4), (698, 6), (701, 2), (702, 3), (707, 3), (708, 2), (709, 4), (711, 10), (712, 2), (713, 2), (714, 6), (717, 13), (718, 4), (720, 5), (724, 7), (730, 7), (731, 3), (732, 8), (733, 5), (736, 1), (737, 3), (739, 3), (741, 3), (742, 3), (744, 7), (745, 4), (746, 5), (751, 7), (761, 15), (764, 3), (766, 3), (769, 3), (771, 10), (772, 4), (774, 11), (776, 13), (777, 7), (780, 4), (783, 9), (784, 1), (785, 4), (786, 4), (788, 1), (790, 3), (792, 7), (794, 4), (797, 3), (799, 3), (805, 3), (807, 3), (809, 12), (811, 5), (815, 4), (816, 3), (818, 3), (824, 2), (827, 3), (830, 3), (831, 3), (832, 3), (833, 6), (834, 3), (835, 4), (836, 5), (837, 6), (838, 3), (839, 6), (840, 15), (841, 4), (842, 3), (844, 7), (845, 6), (846, 3), (847, 6), (848, 16), (849, 4), (851, 5), (852, 8), (853, 5), (855, 9), (857, 7), (858, 4), (859, 5), (860, 4), (861, 4), (862, 5), (863, 5), (864, 3), (865, 3), (866, 4), (867, 4), (868, 4), (869, 5), (870, 5), (871, 6), (872, 7), (873, 8), (874, 6), (875, 3), (876, 5), (877, 5), (878, 8), (879, 5), (880, 5), (881, 3), (882, 13), (883, 3), (884, 3), (885, 4), (886, 3), (887, 2), (888, 4), (889, 3), (890, 7), (891, 3), (892, 6), (893, 3), (894, 7), (895, 8), (896, 3), (897, 4), (898, 13), (900, 9), (901, 4), (902, 4), (903, 12), (904, 4), (905, 3), (906, 10), (907, 4), (908, 3), (909, 4), (910, 5), (911, 3), (912, 5), (913, 4), (914, 3), (915, 3), (916, 5), (918, 3), (919, 6), (920, 4), (921, 7), (922, 4), (923, 5), (924, 4), (925, 10), (926, 5), (928, 5), (930, 4), (931, 8), (932, 5), (933, 6), (934, 4), (935, 5), (937, 3), (939, 6), (941, 3), (942, 5), (943, 5), (944, 6), (945, 6), (946, 8), (947, 9), (948, 7), (949, 4), (950, 3), (952, 4), (954, 4), (955, 4), (956, 3), (957, 6), (958, 4), (959, 4), (960, 11), (961, 6), (962, 3), (963, 5), (964, 3), (965, 3), (966, 4), (967, 5), (968, 5), (969, 3), (970, 3), (971, 3), (972, 4), (973, 7), (974, 4), (976, 5), (977, 3), (978, 5), (980, 4), (981, 4), (982, 2), (983, 3), (984, 3), (985, 6), (986, 5), (987, 12), (989, 3), (990, 13), (992, 5), (993, 5), (994, 6), (995, 9), (996, 3), (997, 5), (998, 10), (999, 5)]

import random
# tests = random.sample(correct_list, 10)
tests = correct_list[2:3]
# tests = [(7, 8)]

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


def create_hand_coded_example(vocab_size: int, seq_len: int, puzzle_id: int = 0, idx = None):
    """Create a hand-coded example sequence for testing"""
    
    # Create input sequence
    if idx is not None:
        all_inputs = np.load("/cephfs/renzirui/projects/HRM-ds/data/sudoku-extreme-1k-aug-1000/test/all__inputs.npy")
        all_labels = np.load("/cephfs/renzirui/projects/HRM-ds/data/sudoku-extreme-1k-aug-1000/test/all__labels.npy")
        inputs = all_inputs[idx]
        labels = all_labels[idx]
        # inputs = np.array([x for x in labels])
        # for i in range(9):
        #     inputs[i] = 1
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

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Carry, HierarchicalReasoningModel_ACTV1InnerCarry
def forward_single_example(model, batch: Dict[str, torch.Tensor], success_steps, noise, max_steps: int = 1000):
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
                if step == success_steps-1:
                    z_H_center = carry.inner_carry.z_H.clone().detach()
                if step == success_steps-1:
                    carry.inner_carry.z_H.add_(noise)
                    carry.steps=torch.ones((1, ), dtype=torch.int32).to("cuda")
                if step == success_steps+1:
                    final_z_H = carry.inner_carry.z_H.clone().detach()
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
        "z_H_center": z_H_center,
        "final_z_H": final_z_H
    }

from sklearn.decomposition import PCA
def fit_pca(all_zHs_list):
    flats = []
    for z_H in all_zHs_list:
        flats.append(z_H.flatten().cpu().float().numpy())
    flat_mat = np.stack(flats)
    pca = PCA(n_components=2)
    pca.fit(flat_mat)
    print(f"[fit_pca] 样本数 {flat_mat.shape[0]}，解释方差 {pca.explained_variance_ratio_}")
    return pca

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
        for idx, success_steps in tests:
            print(f'=====Testing sample {idx}=====')
            print(f'Noise added after step {success_steps}.')

            # Create hand-coded example
            batch = create_hand_coded_example(
                vocab_size=train_metadata.vocab_size,
                seq_len=train_metadata.seq_len,
                puzzle_id=0,
                idx=idx
            )
            inputs = batch["inputs"].squeeze(0)

            results = forward_single_example(
                model=train_state.model,
                batch=batch,
                max_steps=10000,
                noise=0,
                success_steps=success_steps
            )
            _, _, all_zH_list = print_results(batch, results, trace=True)
            z_H_center = results["z_H_center"]

            # For Sample 8 Only
            # z_H_fake = all_zH_list[6]


            def forward_z(model, z_H): #z_H: (82, 512)
                logits = model.model.inner.lm_head(z_H.to("cuda"))[1:] # (81, 11)
                return logits
            
            def differentiable_conflict_loss(board):
                """可微分的冲突损失函数"""
                loss = 0.0
                
                # 将board转换为one-hot编码 (9x9x9)
                board_one_hot = torch.zeros(9, 9, 9)
                for i in range(9):
                    for j in range(9):
                        num = int(board[i, j].item()) - 1  # 转换为0-8
                        if num >= 0:
                            board_one_hot[i, j, num] = 1.0
                
                # 行冲突损失
                for i in range(9):
                    row_sum = torch.sum(board_one_hot[i], dim=0)  # 每行中每个数字的总出现次数
                    # 惩罚重复：sum(max(0, count-1)) 的可微分近似
                    row_excess = torch.relu(row_sum - 1.0)
                    loss += torch.sum(row_excess)
                
                # 列冲突损失
                for j in range(9):
                    col_sum = torch.sum(board_one_hot[:, j], dim=0)
                    col_excess = torch.relu(col_sum - 1.0)
                    loss += torch.sum(col_excess)
                
                # 宫冲突损失
                for box_i in range(3):
                    for box_j in range(3):
                        box_sum = torch.sum(board_one_hot[3*box_i:3*box_i+3, 3*box_j:3*box_j+3], dim=(0, 1))
                        box_excess = torch.relu(box_sum - 1.0)
                        loss += torch.sum(box_excess)
                
                return loss
            
            # interp_error = []
            # for i in range(101):
            #     k = i*0.01
            #     z = (k*z_H_center + (1-k)*z_H_fake).squeeze(0)
            #     logits = forward_z(train_state.model, z)
            #     preds = torch.argmax(logits, dim=-1).view(9,9)-1
            #     # print(f"{k}:\n{preds}")
            #     interp_error.append(differentiable_conflict_loss(preds))
            # plt.figure(1, (8, 6))
            # plt.plot([(101-i)*0.01 for i in range(101)], interp_error, color='red')
            # plt.savefig("interp.png")
            # plt.close()

            pca = fit_pca(all_zH_list)

            # # ans = {(0, 0): 0}
            ans = {}
            dest = {}
            scale = ((82*512)**0.5)/32


            for i in range(-10, 30+1):
                for j in range(-15, 15+1):
                    print(f'noise {i}, {j}: ', end="")
                    d1 = torch.from_numpy(pca.components_[0]*scale).view(1, 82, 512).to("cuda")
                    d2 = torch.from_numpy(pca.components_[1]*scale).view(1, 82, 512).to("cuda")
                    noise = i * d1 + j * d2
                    
                    logits = forward_z(train_state.model, (z_H_center+noise).squeeze(0))
                    preds = torch.argmax(logits, dim=-1).view(9,9)-1
                    ans[(i, j)] = differentiable_conflict_loss(preds)

                    # # Forward the example
                    # results = forward_single_example(
                    #     model=train_state.model,
                    #     batch=batch,
                    #     max_steps=10000,
                    #     noise=noise,
                    #     success_steps=success_steps
                    # )
                    # final_z = results["final_z_H"]

                    # _, restore, _ = print_results(batch, results)
                    # dest[(i, j)] = (pca.transform(final_z.view(1, -1).float().cpu())-pca.transform(z_H_center.view(1, -1).float().cpu()))[0]/scale

            # plot_ij_dict(ans, [(pca.transform(z.view(1, -1).float().cpu())-pca.transform(z_H_center.view(1, -1).float().cpu()))[0]/scale for z in all_zH_list[1:]], idx, dest)
            plot_ij_dict_3d(ans, [], idx, dest)
           
        
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

python single_example_noise_interp.py --checkpoint_path "/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 pastel-lorikeet/step_26040" > single_inference_result.txt

python single_example_noise.py --checkpoint_path "/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000-hint-rand-all ACT-torch/HierarchicalReasoningModel_ACTV1 burrowing-pudu/step_52080" > single_inference_result.txt

python single_example_noise.py --checkpoint_path "/cephfs/renzirui/projects/HRM-ds/checkpoints/Sudoku-extreme-1k-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 optimal-mongrel" > single_inference_result.txt

'''
