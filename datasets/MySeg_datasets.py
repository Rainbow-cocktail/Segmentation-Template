from torch.utils.data import Dataset
import os
import numpy as np
import tifffile
import torch
import PIL.Image as Image


class MyDataset(Dataset):
    """
    高光谱图像分割数据集加载类

    参数:
    ---------
    data_dir : str
        数据集的根目录，包含 images/ 和 masks/ 文件夹，以及 train.txt/valid.txt 等分割文件
    collection : str
        图像模态类型，支持 'rgb'、'vis'、'nir' 等（将作为文件名中间部分）
    classes : list or tuple, optional
        类别名称列表，例如 ['background', 'road']，默认仅用于记录总类别数
    mode : str, optional
        模式选择：'train' 或 'valid'，控制读取哪个 split 文件
    """

    def __init__(self, data_dir, collection, classes=None, mode='train', **kwargs):
        self.data_dir = data_dir
        self.collection = collection.lower()  # 模态名称统一小写
        self.mode = mode.lower()

        # 加载图像名称列表（如 ['0001', '0002', ...]）
        txt_file = os.path.join(data_dir, f'{self.mode}.txt')
        self.name_list = np.genfromtxt(txt_file, dtype=str)

        # 类别索引（例如 [0, 1]）
        self.classes = list(range(len(classes))) if classes else None

    def __getitem__(self, idx):
        """
        返回第 idx 个样本的数据
        ----------
        输出:
        image : torch.Tensor, shape [C, H, W], float32, 已归一化到 [0, 1]
        mask  : torch.Tensor, shape [H, W], int64, 表示每个像素的类别索引
        """
        name = f'{self.name_list[idx]}_{self.collection}.tif'
        image_path = os.path.join(self.data_dir, 'images', name)
        mask_path = os.path.join(self.data_dir, 'masks', name)

        # 读取图像和掩码，tif 格式，高光谱图像为 [C, H, W]
        image = tifffile.imread(image_path).astype(np.float32) / 255.0  # 归一化
        mask = tifffile.imread(mask_path).astype(np.int64)

        # 转为 torch.Tensor 格式
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        """
        返回数据集中样本数量
        """
        return len(self.name_list)
