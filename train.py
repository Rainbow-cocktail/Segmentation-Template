import torch

torch.set_float32_matmul_precision('high')
from argparse import ArgumentParser
import numpy as np
import os
import yaml
from datetime import datetime
from pathlib import Path
import shutil

from framework import LightningSeg
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


def main(options):
    # 读取 YAML 配置文件
    with open(options.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 保证实验可复现
    SEED = config.get('SEED', 1234)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 初始化 Lightning 模型
    model = LightningSeg(model_params=config['model_cfgs'],
                         dataset_params=config['dataset_cfgs'],
                         loss_params=config['loss_cfgs'],
                         train_params=config['train_cfgs'])


    # 定义实验版本名称
    version_name = datetime.now().strftime("run_%Y%m%d-%H%M%S")
    model_name = model.model.name

    # 输出目录
    output_dir = Path("outputs") / model_name / version_name
    log_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    test_dir = output_dir / "test"
    for d in [log_dir, ckpt_dir, results_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 保存路径传入模型，便于 framework.py 中使用
    model.output_dir = output_dir
    model.results_dir = results_dir
    model.test_dir = test_dir

    # 复制配置文件以保留实验元信息
    shutil.copy(options.config, output_dir / "config.yaml")

    # 定义模型检查点回调（保留 val_mIoU 指标最优模型）
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{model_name}--{{epoch:02d}}-{{val_mIoU:.4f}}",
        monitor="val_mIoU",
        save_top_k=1,
        mode="max",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )

    # 添加 Logger：TensorBoard + CSV，统一 version 和路径
    logger_tb = TensorBoardLogger(save_dir=log_dir.parent, name="logs", version=version_name)
    logger_csv = CSVLogger(save_dir=log_dir.parent, name="logs", version=version_name)

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
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()

    # model and training setup
    parser.add_argument('--config', type=str, help='name a yaml running configuration')
    option = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(option)
