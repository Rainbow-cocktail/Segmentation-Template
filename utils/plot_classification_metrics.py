import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
import os


def plot_multiclass_metrics(y_true, y_pred, class_names=None, exclude_class_0=True, save_dir=None):
    """
    绘制多分类任务中的 ROC、PR、F1 vs Threshold 三类曲线。
    对于每个类别均需要绘制 ROC 曲线、PR 曲线、F1 vs Threshold 曲线。
    对于背景类（通常为第 0 类）不绘制 ROC 曲线，也可以绘制。
    这个函数可以单独调用，也可以在训练完成后调用。

    参数：
    - y_true: [N]，每个像素的真实标签（整数）
    - y_pred: [N, C]，每个像素对每个类别的预测概率, 已经经过softmax处理
    - class_names: List[str]，类别名称列表（可选）
    - exclude_class_0: 是否跳过第 0 类（通常为背景）
    - save_dir: 可选，图像保存目录；若为 None 则仅显示图像

    使用说明：
    ---------------------------
    你可以在训练完成后单独在 Notebook 或脚本中调用该函数，读取保存的 .npz 文件并绘图。

    示例用法：
    >>> import numpy as np
    >>> data = np.load("outputs/your_model/run_xxxx/results/classification_data_epoch30.npz")
    >>> y_true = data["y_true"]
    >>> y_pred = data["y_pred"]
    >>> class_names = ["背景", "道路", "建筑", "车", "人"]
    >>> plot_multiclass_metrics(y_true, y_pred, class_names=class_names, exclude_class_0=True)
    """
    num_classes = y_pred.shape[1]  # 类别总数
    y_true_bin = np.eye(num_classes)[y_true]  # 转为 one-hot 编码 shape: (N, C)

    # 确定要处理的类别（是否排除背景类）
    class_indices = range(1, num_classes) if exclude_class_0 else range(num_classes)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # ---------- ROC 曲线 ----------
    plt.figure(figsize=(6, 5))
    for i in class_indices:
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {score:.2f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        roc_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(roc_path)
        print(f"[Saved] ROC curve to {roc_path}")
    else:
        plt.show()

    # ---------- PR 曲线 ----------
    plt.figure(figsize=(6, 5))
    for i in class_indices:
        try:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_pred[:, i])
            plt.plot(recall, precision, label=f"{class_names[i]} (AP = {ap:.2f})")
        except Exception:
            continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        pr_path = os.path.join(save_dir, "pr_curve.png")
        plt.savefig(pr_path)
        print(f"[Saved] PR curve to {pr_path}")
    else:
        plt.show()

    # ---------- F1 vs Threshold 曲线 ----------
    thresholds = np.linspace(0.0, 1.0, 50)  # 遍历 50 个阈值
    plt.figure(figsize=(6, 5))
    for i in class_indices:
        try:
            f1s = []
            for t in thresholds:
                pred_bin = (y_pred[:, i] >= t).astype(int)  # 概率二值化
                f1 = f1_score(y_true_bin[:, i], pred_bin, zero_division=0)
                f1s.append(f1)
            plt.plot(thresholds, f1s, label=f"{class_names[i]}")
        except Exception:
            continue
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        f1_path = os.path.join(save_dir, "f1_vs_threshold.png")
        plt.savefig(f1_path)
        print(f"[Saved] F1 vs Threshold curve to {f1_path}")
    else:
        plt.show()


def plot_loss_curve(train_loss_list, val_loss_list=None, save_path=None, title="Loss Curve"):
    """
    绘制训练和验证的 Loss 曲线

    参数:
    - train_loss_list: List[float]，每个 epoch 的训练 loss
    - val_loss_list: List[float]，每个 epoch 的验证 loss（可选）
    - save_path: str，保存路径
    - title: str，图表标题
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_list, label='Train Loss', linewidth=2)

    if val_loss_list:
        plt.plot(val_loss_list, label='Val Loss', linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Loss 曲线已保存到: {save_path}")

    plt.close()