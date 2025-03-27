import torch

torch.set_float32_matmul_precision('high')
from argparse import ArgumentParser
import numpy as np
import os
import yaml

from framework import LightningSeg
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


def main(options):
    # 读取 YAML 配置文件
    with open(options.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 初始化 Lightning 模型
    model = LightningSeg(model_params=config['model_cfgs'],
                         dataset_params=config['dataset_cfgs'],
                         loss_params=config['loss_cfgs'],
                         train_params=config['train_cfgs'])


    # 定义模型检查点回调（保留 val_mIoU 指标最优模型）
    # 保存模型的设置
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{model.model.name}--{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        save_top_k=1,
        mode="max",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )

    # 添加 Logger：TensorBoard + CSV
    logger_tb = TensorBoardLogger(save_dir="logs", name=model.model.name)
    logger_csv = CSVLogger(save_dir="logs", name=model.model.name)

    device = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        devices=1,
        accelerator=device,
        max_epochs=config['train_cfgs']['epochs'],
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stopping],
        logger=[logger_tb, logger_csv]
    )

    trainer.fit(model)


if __name__ == '__main__':
    # 保证实验可复现
    SEED = 2334
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()

    # model and training setup
    parser.add_argument('--config', type=str, help='name a yaml running configuration')
    option = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(option)
