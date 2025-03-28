#  Muti-channel Image Segmentation

本仓库更多是对[NUST-Machine-Intelligence-Laboratory/hsi_road](https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road)仓库的拓展，使之适配更多网络支持多类分割评估指标、混淆矩阵与多类别 ROC/PR/F1 曲线可视化。

本项目在以下开源项目基础上扩展与重构而来，具体引用详见文末 Reference。

---

## 文件结构

```bash
HSI_segmentation_template/
├── configs/
│   └── unet-mobilenetv3.yaml               # 配置文件
├── datasets/
│   ├── roadSeg_datasets.py                # 原始高光谱数据集
│   ├── MySeg_datasets.py                  # 你后续自定义的数据集（如需要）
│   └── __init__.py                         # 数据集注册入口
├── segmentations/
│   ├── common.py                           # 通用模型结构
│   ├── fcn.py                              # FCN 分割头
│   ├── pspnet.py                           # PSPNet
│   ├── refinenet.py                        # RefineNet
│   ├── segformer.py                        # SegFormer（MIT专用）
│   ├── unet.py                             # UNet
│   └── __init__.py                         # 模型注册
├── utils/
│   ├── classification_export.py            # 导出 softmax+标签 的 npz
│   ├── losses.py                           # 各类损失函数
│   ├── metrics_ext.py                      # 混淆矩阵 + 多指标计算
│   ├── optimizers.py                       # 优化器与调度器管理
│   ├── plot_classification_metrics.py      # 绘制 ROC/PR/F1 曲线
│   └── __init__.py
├── framework.py                            # Lightning 模型主框架
├── train.py                                # 训练入口
├── LICENSE                                 # Apache 2.0 License
├── README.md                               # 项目说明文档
├── requirements.txt                        # 依赖列表
└── outputs/                                # 自动生成的结果文件
```

---

## 🔺 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
### 2. 准备数据集
支持任意通道图像， 推荐使用 [NUST-Machine-Intelligence-Laboratory/hsi_road](https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road)

### 3. 训练模型
```bash
python train.py --config your_config_file.yaml
```

训练结束后，会在 `outputs/` 下生成：
- `checkpoints/`：val_mIoU 最优模型
- `results/`：混淆矩阵图 + 多分类曲线图（ROC、PR、F1）
- `logs/`：TensorBoard + CSV 日志

### 4. 查看日志

使用tensorboard正常查看即可 

```bash
tensorboard --logdir outputs/unet/logs
```

---

## 💡 支持结构

### ✅ 支持编码器（encoder_name）

- `mobilenetv3`, `resnet`, `resnext`, `senet`, `vgg`, `mit`（SegFormer 专用）

### ✅ 支持分割头（model_name）

- `unet`, `fcn`, `pspnet`, `refinenet`, `segformer`

> 注意：SegFormer 解码器必须与 `mit` 编码器配合使用。

---

## 🌍 配置文件说明（unet-mobilenetv3.yaml）

```yaml
SEED: 0 # 全局随机种子

model_cfgs:
  model_name: unet  # 选择分割头
  model_args:
    encoder_name: mobilenetv3 # 选择分类头
    in_channel_nb: 25 # 输入通道
    classes_nb: 2 # 分类数

dataset_cfgs:
  name: hsi_road # 数据集名称
  data_dir: F:\Data\hsi_road\hsi_road # 数据集根目录
  collection: nir # nir rgb
  classes:  # 分类名称
    - background
    - road

loss_cfgs:
  loss_name: ce
  loss_args:
    alpha: 0.75         # used by focal, focal_multi, combo
    gamma: 2.0         # used by focal, focal_multi
    reduction: mean    # used by ce, bce, focal, focal_multi
    weight: ~          # used by ce, bce
    smooth: 1e-6        # used by dice

train_cfgs:
  optimizer: adam               # 可选项：adam / adamw / sgd
  lr_rate: 0.01
  lr_scheduler: step             # 可选项：step / multistep / cosine
  lr_scheduler_step: 10
  lr_scheduler_gamma: 0.1
  lr_scheduler_milestones: [ 30, 60 ]  # 如果用 multistep
  lr_scheduler_tmax: 50              # 如果用 cosine
  batch_size: 4
  epochs: 7
  save_cm_interval: 5

plot_cfgs:
  save_last_epoch_result: True
  plot_classification_curve: True
  max_points: 100000  # 最多保存的像素点数
```

---

## 📊 评估指标
- Pixel Accuracy
- Mean IoU
- Precision / Recall / F1 Score
- Confusion Matrix（支持 TensorBoard + 本地保存）
- ROC / PR / F1-Threshold 曲线（多分类）

---

## 🌟 TODO
- [ ] 重构为 LightningDataModule

---

## 🛠️自定义项目

本仓库的核心价值在于：提供了**常用的分割网络结构**和**节省时间的可视化与评估模块**，适合快速适配到你自己的语义分割任务中。

若你希望使用自己的数据，只需参考 `datasets/roadSeg_datasets.py` 的模板，实现一个新的 `Dataset` 类，并在 `datasets/__init__.py` 中注册即可。

> 建议输入图像的尺寸不小于 **224×224**，以确保网络结构和 patch 操作的有效性。

若你希望添加新的模型或修改现有结构，可按如下步骤操作：

1. **添加编码器（Encoder）**
    在 `segmentations/encoders/` 中新增编码器定义，并在 `segmentations/encoders/__init__.py` 中进行注册。

2. **添加解码器（Decoder）或整合模型结构**
    在 `segmentations/` 目录下创建新的文件（如 `my_decoder.py`），实现你自己的分割结构，并在 `segmentations/__init__.py` 中注册。

---

## 📃 Reference and Acknowledgements

the codes are copied heavily from the following repository:

- 【HSI Road Dataset】https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road
- 【SegFormer Pytorch】https://github.com/bubbliiiing/segformer-pytorch
- 【Awesome Semantic Segmentation】https://github.com/Tramac/awesome-semantic-segmentation-pytorch

---

## LICENSE

本项目采用 **Apache License 2.0** 协议，详见 LICENSE 文件。
