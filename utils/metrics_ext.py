import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import os


class SegmentationMetric:
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = None

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        :param preds: [B, C, H, W] logits or probs
        :param targets: [B, H, W] ground truth class indices
        """
        if preds.shape[1] > 1:
            preds = torch.argmax(preds, dim=1)
        else:
            preds = (torch.sigmoid(preds) > 0.5).long().squeeze(1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        cm = torch.bincount(
            self.num_classes * targets + preds,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros(
                (self.num_classes, self.num_classes), dtype=torch.int64
            ).type_as(cm)  # .to(cm.device) is also ok

        self.confusion_matrix += cm

    def get_scores(self):
        cm = self.confusion_matrix.float()
        TP = torch.diag(cm)
        FP = cm.sum(0) - TP
        FN = cm.sum(1) - TP
        TN = cm.sum() - (TP + FP + FN)

        pixel_acc = TP.sum() / cm.sum()
        class_iou = TP / (TP + FP + FN + 1e-8)
        miou = class_iou.mean()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'Pixel_Acc': pixel_acc.item(),
            'Mean_IoU': miou.item(),
            'Class_IoU': class_iou.tolist(),
            'Precision': precision.tolist(),
            'Recall': recall.tolist(),
            'F1': f1.tolist(),
            'Confusion_Matrix': cm.int().tolist()
        }

    def reset(self):
        self.confusion_matrix.zero_()

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


if __name__ == '__main__':
    # 创建一个 metric 实例（10 类 + 背景 = 11 类）
    metric = SegmentationMetric(num_classes=11, ignore_index=None)

    # 模拟两个 batch（每个 batch：B=2, C=11, H=4, W=4）
    for i in range(2):  # 2 个 batch
        logits = torch.randn([2, 11, 4, 4])  # 模型输出（raw logits）
        labels = torch.randint(0, 11, (2, 4, 4))  # 标签

        metric.update(logits, labels)

    # 获取结果
    scores = metric.get_scores()
    print("✅ 测试通过，结果如下：")
    for k, v in scores.items():
        if k != "Confusion_Matrix":
            print(f"{k}: {v}")

    # 可视化混淆矩阵
    class_names = [f"class{i}" for i in range(11)]
    os.makedirs("test_output", exist_ok=True)
    metric.plot_confusion_matrix(class_names, save_path="test_output/confusion_matrix.png")

    # TensorBoard 写入（可选）
    writer = SummaryWriter(log_dir="test_output/tensorboard")
    metric.plot_confusion_matrix(class_names, writer=writer, global_step=0)
    writer.close()
    print("✅ 混淆矩阵图已保存至 test_output/ 下")
