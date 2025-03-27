import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os


class SegmentationMetric:
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.outputs = []  # 存 softmax scores
        self.targets = []  # 存 ground truth label

    def update(self, preds_raw: torch.Tensor, targets: torch.Tensor):
        """
        更新语义分割评估指标，包含混淆矩阵统计与后续可视化所需数据的积累。

        主要功能：
        1. 根据预测结果与真实标签，统计混淆矩阵（Confusion Matrix）；
        2. 收集每个 batch 的 softmax 概率（用于绘制 ROC/PR/F1 曲线）；
        3. 累积每个 batch 的 targets。

        参数说明：
        -----------
        preds_raw : torch.Tensor
            模型输出的原始 logits，形状为 [B, C, H, W]，其中：
            - B: 批次大小
            - C: 类别数
            - H, W: 图像高宽

        targets : torch.Tensor
            ground truth 标签，形状为 [B, H, W]，其中每个像素的值 ∈ [0, C-1]
        """
        if preds_raw.device != targets.device:
            targets = targets.to(preds_raw.device)

        # 用于计算混淆矩阵
        preds = torch.argmax(preds_raw, dim=1).view(-1)  # [B,NC,H,W] -> [B, H, W] -> [B * H * W]
        targets_flat = targets.view(-1)  # [B * H * W]

        mask = (targets_flat != self.ignore_index) if self.ignore_index is not None else torch.ones_like(targets_flat,
                                                                                                         dtype=torch.bool)
        preds = preds[mask]
        targets_flat = targets_flat[mask]

        cm = torch.bincount(
            self.num_classes * targets_flat + preds,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix = self.confusion_matrix.to(cm.device)
        self.confusion_matrix += cm

        # 保存预测概率和标签用于后续曲线绘制
        probs = torch.softmax(preds_raw, dim=1).detach().cpu()  # [B, C, H, W]
        self.outputs.append(probs)
        self.targets.append(targets.detach().cpu())

    def get_scores(self):
        """
        计算并返回语义分割评估指标，包含：
        - Pixel Accuracy: 所有像素预测对的比例（准确率）→ 容易被背景类拉高
        - Mean IoU: 所有类别 IoU 平均值 → 语义分割主指标
        - Precision: 所有类 precision 平均 → 预测正确率
        - Recall:  所有类 recall 平均 → 召回率
        - F1: 精准率与召回率的加权平均 → 权衡指标
        - IoU per class: 每个类的 IoU 分数（list）→ 方便看哪个类难预测
        """
        cm = self.confusion_matrix.float()
        TP = torch.diag(cm)  # shape: [C] -> 每个类别的 True Positive 数
        precision = TP / (cm.sum(dim=0) + 1e-8)  # shape: [C]
        recall = TP / (cm.sum(dim=1) + 1e-8)  # shape: [C]
        f1 = 2 * precision * recall / (precision + recall + 1e-8)  # shape: [C]
        iou = TP / (cm.sum(dim=1) + cm.sum(dim=0) - TP + 1e-8)  # shape: [C]

        pixel_acc = TP.sum() / cm.sum()  # scaler
        mean_iou = iou.mean()  # scaler

        return {
            'Pixel_Acc': pixel_acc.item(),
            'Mean_IoU': mean_iou.item(),
            'Precision': precision.mean().item(),
            'Recall': recall.mean().item(),
            'F1': f1.mean().item(),
            'IoU_per_class': iou.tolist()
        }

    def reset(self):
        self.confusion_matrix.zero_()
        self.outputs.clear()
        self.targets.clear()

    def plot_confusion_matrix(self, class_names, save_path=None, writer: SummaryWriter = None,
                              global_step=None):
        cm = self.confusion_matrix.cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(len(class_names)),
               yticks=np.arange(len(class_names)),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True label',
               xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if writer and global_step is not None:
            writer.add_figure("Confusion_Matrix", fig, global_step=global_step)
        plt.close(fig)


