from torch.utils.data import Dataset
import os
import numpy as np
import tifffile
import torch
import PIL.Image as Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BioSeg_RGB_datasets(Dataset):
    """
    高光谱图像分割数据集加载类

    参数:
    ---------
    data_dir : str
        数据集的根目录，包含 images/ 和 masks/ 文件夹，以及 train.txt/valid.txt 等分割文件
    classes : list or tuple, optional
        类别名称列表，例如 ['background', 'road']，默认仅用于记录总类别数
    mode : str, optional
        模式选择：'train' 或 'valid'，控制读取哪个 split 文件
    transform: callable, optional
        是否运用数据增强，默认 None， 可选 'online' 或其他自定义函数
    """

    def __init__(self, data_dir, collection='nir', classes=None, mode='train', transform=None, **kwargs):
        self.data_dir = data_dir
        self.collection = collection.lower()  # 模态名称统一小写
        self.mode = mode.lower()

        if transform is None:
            self.transform = None
        elif transform == 'online':
            if mode == 'train':
                self.transform = self.online_transform()
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=[0.5] * 7, std=[0.5] * 7),
                    ToTensorV2()
                ])
        else:
            self.transform = transform

        # 加载图像名称列表
        txt_file = os.path.join(data_dir, f'{self.mode}.txt')
        self.name_list = np.genfromtxt(txt_file, dtype=str)

        # 类别索引， 在demo任务中， 只有 0 1 2 三类 ， 无stain
        self.classes = list(range(len(classes))) if classes else None

        self.image_dir = os.path.join(self.data_dir, 'images')
        self.mask_dir = os.path.join(self.data_dir, 'masks')

    def __getitem__(self, idx):
        """
        返回第 idx 个样本的数据
        ----------
        输出:
        image : torch.Tensor, shape [C, H, W], float32, 已归一化到 [0, 1]
        mask  : torch.Tensor, shape [H, W], int64, 表示每个像素的类别索引
        """
        prefix = self.name_list[idx]
        # 读取RGB图像（3通道，uint8）
        rgb_path = os.path.join(self.image_dir, prefix + '_RGB.tif')
        rgb = tifffile.imread(rgb_path).astype(np.float32) / 255.0  # 归一化 [0, 1]
        if rgb.shape[2] != 3:
            raise ValueError(f"RGB 图像格式错误: {rgb_path}")

        # 拼接所有波段：RGB(3) + 其他(4) → (H, W, 7)
        image = rgb

        # 读取mask
        mask_path = os.path.join(self.mask_dir, prefix + '_RGB.png')
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.int64)

        # ==== 在线数据增强 ====
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            # Albumentations 保证了同步处理图像和 mask
            image = augmented['image']  # 已是Tensor
            mask = augmented['mask']  # 已是Tensor
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # → (3, H, W)
            mask = torch.from_numpy(mask).long()  # → (H, W)

        return image, mask

    def __len__(self):
        """
        返回数据集中样本数量
        """
        return len(self.name_list)

    @ staticmethod
    def online_transform():
        """
        对图像和掩膜进行数据增强, 增强统一用这个包
        ----------
        输出:
        image : torch.Tensor, shape [C, H, W], float32
        mask  : torch.Tensor, shape [H, W], int64
        """
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.5] * 7, std=[0.5] * 7),  # 简单标准化
            ToTensorV2()
        ])
        return aug


if __name__ == '__main__':
    # 测试数据集类
    data_dir = r"F:\Data\bio_colonization"
    dataset = BioSeg_datasets(data_dir, collection='RGB', mode='train')
    print(f"数据集大小: {len(dataset)}")
    for i in range(5):
        image, mask = dataset[i]
        print(f"样本 {i}: 图像形状 {image.shape}, 掩膜形状 {mask.shape}")
