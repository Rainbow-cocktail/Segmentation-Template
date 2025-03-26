import lightning.pytorch as pl

import numpy as np
import imageio

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset
from segmentations import get_model
from utils import get_loss, IoUMetric, get_optimizer_and_scheduler


def infer_activation_from_loss(loss_name):
    if loss_name in ['ce', 'dice', 'combo', 'focal_multi']:
        return 'softmax'
    elif loss_name in ['bce', 'focal']:
        return 'sigmoid'
    else:
        return None


class LightningSeg(pl.LightningModule):

    def __init__(self, model_params, dataset_params, loss_params, train_params):
        super(LightningSeg, self).__init__()

        self.model_cfgs = model_params
        self.dataset_cfgs = dataset_params
        self.loss_cfgs = loss_params
        self.train_cfgs = train_params

        # 加载模型
        self.model = get_model(self.model_cfgs['model_name'], self.model_cfgs['model_args'])
        # 加载损失函数
        self.loss = get_loss(self.loss_cfgs['loss_name'], self.loss_cfgs['loss_args'])
        # 计算 IoU 评价指标
        loss_name = self.loss_cfgs['loss_name']
        self.metric = IoUMetric(activation=infer_activation_from_loss(loss_name))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        iou = self.metric(y_hat, y)[1]  # 计算 IoU 指标

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_iou", iou, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        iou = self.metric(y_hat, y)[1]

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_iou", iou, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        iou = self.metric(y_hat, y)[1]

        y = torch.squeeze(y, dim=0).cpu().numpy()[1, ...].astype(np.uint8) * 128
        y_hat = torch.squeeze(torch.argmax(y_hat, dim=1), dim=0).cpu().numpy().astype(np.uint8) * 128
        z = np.zeros_like(y, dtype=y.dtype)
        im = np.stack([y, y_hat, z], axis=-1)

        imageio.imwrite(f'test/{batch_idx}.png', im)

        return {"test_iou": iou}

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
