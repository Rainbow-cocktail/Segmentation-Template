import torch
import numpy as np
from pathlib import Path


def export_classification_data(outputs, targets, save_path: Path, num_classes: int, max_points: int = None):
    """
    从累计的 outputs 和 targets 中提取概率和标签，保存为 .npz 文件。

    参数：
    - outputs: List[Tensor], 每个 [B, C, H, W]，softmax 后的概率
    - targets: List[Tensor], 每个 [B, H, W]，整数型标签
    - save_path: Path，保存路径
    - num_classes: int，类别数
    - max_points: int，可选，最多保留多少个像素点（避免内存爆炸）

    输出：
    - 保存一个 .npz 文件，包含：
        - y_true: [N]，所有像素标签
        - y_pred: [N, C]，每个像素的 softmax 概率
    """
    y_pred_all = torch.cat(outputs, dim=0)  # [B_total, C, H, W]
    y_true_all = torch.cat(targets, dim=0)  # [B_total, H, W]

    # 展平
    y_pred_flat = y_pred_all.permute(0, 2, 3, 1).reshape(-1, num_classes)  # [N, C]
    y_true_flat = y_true_all.reshape(-1)  # [N]

    if max_points is not None and y_pred_flat.shape[0] > max_points:
        # 只保留前 max_points 个
        y_pred_flat = y_pred_flat[:max_points]
        y_true_flat = y_true_flat[:max_points]

    y_pred_np = y_pred_flat.numpy()
    y_true_np = y_true_flat.numpy()

    np.savez(save_path, y_true=y_true_np, y_pred=y_pred_np)
    print(f"[INFO] 保存分类数据到 {save_path}, 共 {y_pred_np.shape[0]} 个像素点")
