import lightning.pytorch as pl

import numpy as np
import imageio
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset
from segmentations import get_model
from utils import get_loss, get_optimizer_and_scheduler, SegmentationMetric, export_classification_data


def infer_activation_from_loss(loss_name):
    if loss_name in ['ce', 'dice', 'combo', 'focal_multi']:
        return 'softmax'
    elif loss_name in ['bce', 'focal']:
        return 'sigmoid'
    else:
        return None


class LightningSeg(pl.LightningModule):

    def __init__(self, model_params, dataset_params, loss_params, train_params, **kwargs):
        super(LightningSeg, self).__init__()

        self.model_cfgs = model_params
        self.dataset_cfgs = dataset_params
        self.loss_cfgs = loss_params
        self.train_cfgs = train_params
        self.plot_cfgs = kwargs.get('plot_cfgs') or {}

        self.model = get_model(self.model_cfgs['model_name'], self.model_cfgs['model_args'])
        self.loss = get_loss(self.loss_cfgs['loss_name'], self.loss_cfgs['loss_args'])
        self.seg_metric = SegmentationMetric(num_classes=self.model_cfgs['model_args']['classes_nb'], ignore_index=None)
        self.save_cm_interval = self.train_cfgs.get('save_cm_interval', 10)

        # 设置输出目录
        self.output_dir = Path(getattr(self, "output_dir", "outputs"))
        self.results_dir = Path(getattr(self, "results_dir", "results"))
        self.test_dir = Path(getattr(self, "test_dir", "test"))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        x, y = batch  # y.shape: B,H,W
        y_hat = self(x)  # B,C,H,W
        loss = self.loss(y_hat, y)
        self.seg_metric.update(y_hat.detach(), y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        scores = self.seg_metric.get_scores()

        # 基础指标记录
        self.log("val_mIoU", scores["Mean_IoU"])
        self.log("val_PixelAcc", scores["Pixel_Acc"])
        self.log("val_Precision", scores["Precision"])
        self.log("val_Recall", scores["Recall"])
        self.log("val_F1", scores["F1"])

        class_names = [str(i) for i in range(self.seg_metric.num_classes)]
        save_dir = self.results_dir
        os.makedirs(save_dir, exist_ok=True)

        epoch = self.current_epoch
        logger = self.logger.experiment if hasattr(self.logger, "experiment") else None

        if ((epoch + 1) % self.save_cm_interval == 0) or (epoch + 1) == self.trainer.max_epochs:
            cm_path = save_dir / f"confusion_matrix_epoch{epoch}.png"
            self.seg_metric.plot_confusion_matrix(class_names, save_path=cm_path) # 全部类别的混淆矩阵（像素点数）
            cm_norm_path = save_dir / f"confusion_matrix_norm_epoch{epoch}.png"
            self.seg_metric.plot_confusion_matrix(class_names, save_path=cm_norm_path, normalize=True) # 正规化后的混淆矩阵（0~1）
            if logger:
                self.seg_metric.plot_confusion_matrix(class_names, writer=logger, global_step=epoch)

            # 写入 training_log.txt
            log_file = save_dir / "training_log.txt"
            train_loss = self.trainer.callback_metrics.get('train_loss', None)
            val_loss = self.trainer.callback_metrics.get('val_loss', None)
            with open(log_file, "a") as f:
                f.write(f"\n[Epoch {epoch}] Evaluation Metrics\n")
                f.write(
                    f"Train Loss      : {train_loss:.4f}\n" if train_loss is not None else "Train Loss      : N/A\n")
                f.write(f"Val Loss        : {val_loss:.4f}\n" if val_loss is not None else "Val Loss        : N/A\n")
                f.write(f"Pixel Accuracy  : {scores['Pixel_Acc']:.4f}\n")
                f.write(f"Mean IoU        : {scores['Mean_IoU']:.4f}\n")
                f.write(f"Precision (avg) : {scores['Precision']:.4f}\n")
                f.write(f"Recall    (avg) : {scores['Recall']:.4f}\n")
                f.write(f"F1        (avg) : {scores['F1']:.4f}\n")
                f.write("Class-wise IoU  : " + ", ".join([f"{v:.4f}" for v in scores["IoU_per_class"]]) + "\n")

        if ((epoch + 1) == self.trainer.max_epochs) and self.plot_cfgs.get("save_last_epoch_result", False):
            export_classification_data(
                outputs=self.seg_metric.outputs,
                targets=self.seg_metric.targets,
                save_path=self.results_dir / f"classification_data_epoch{epoch + 1}.npz",
                num_classes=self.seg_metric.num_classes,
                max_points=self.plot_cfgs.get("max_points", 10000)  # 可选限制：最多保留多少个像素，避免显存爆炸
            )

        self.seg_metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.seg_metric.update(y_hat.detach(), y)

        y = torch.squeeze(y, dim=0).cpu().numpy()[1, ...].astype(np.uint8) * 128
        y_hat = torch.squeeze(torch.argmax(y_hat, dim=1), dim=0).cpu().numpy().astype(np.uint8) * 128
        z = np.zeros_like(y, dtype=y.dtype)
        im = np.stack([y, y_hat, z], axis=-1)

        imageio.imwrite(self.test_dir / f"{batch_idx}.png", im)

        return {}

    def on_test_epoch_end(self):
        scores = self.seg_metric.get_scores()

        self.log("test_mIoU", scores["Mean_IoU"])
        self.log("test_PixelAcc", scores["Pixel_Acc"])

        class_names = [str(i) for i in range(self.seg_metric.num_classes)]
        save_path = self.results_dir / "test_confusion_matrix.png"
        self.seg_metric.plot_confusion_matrix(class_names, save_path=save_path)

        if hasattr(self.logger, "experiment"):
            self.seg_metric.plot_confusion_matrix(class_names, writer=self.logger.experiment,
                                                  global_step=self.current_epoch)


        self.seg_metric.reset()

    def configure_optimizers(self):
        return get_optimizer_and_scheduler(self, self.train_cfgs)

    def train_dataloader(self):
        # 数据加载
        ds = get_dataset(train_mode='train',
                         args=self.dataset_cfgs)
        loader = DataLoader(
            dataset=ds,
            batch_size=self.train_cfgs['batch_size'],
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )
        return loader

    def val_dataloader(self):
        ds = get_dataset('valid', self.dataset_cfgs)
        loader = DataLoader(
            dataset=ds,
            batch_size=self.train_cfgs['batch_size'],
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )
        return loader

    def test_dataloader(self):
        ds = get_dataset('valid', self.dataset_cfgs)
        loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=1)
        return loader
