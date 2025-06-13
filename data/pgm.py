import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset

class PGM(Dataset):
    """
    新版 PGM Dataset，假设每个 .npz 中的 'image' 欄位已经是 (16, image_size, image_size)。
    不再在运行时做 cv2.resize。
    """
    def __init__(
        self, dataset_dir, data_split=None, image_size=80,
        transform=None, subset="None"
    ):

        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform

        # 列出所有符合 "*_{split}_*.npz" 的文件名
        self.file_names = [
            os.path.basename(f)
            for f in glob.glob(os.path.join(self.dataset_dir, "*_" + self.data_split + "_*.npz"))
        ]
        self.file_names.sort()

        # Sanity check
        if subset == 'train':
            assert len(self.file_names) == 1200000, f'Train length = {len(self.file_names)}'
        if subset == 'val':
            assert len(self.file_names) == 20000, f'Validation length = {len(self.file_names)}'
        if subset == 'test':
            assert len(self.file_names) == 200000, f'Test length = {len(self.file_names)}'

    def __len__(self):
        return len(self.file_names)

    def _get_data(self, idx):
        data_file = self.file_names[idx]
        data_path = os.path.join(self.dataset_dir, data_file)
        data = np.load(data_path)

        # 直接取出已经是 (16, image_size, image_size) 的图
        image = data["image"]
        # 最好再做一个断言，确保形状正确
        assert (
            image.ndim == 3 and image.shape[1] == self.image_size and image.shape[2] == self.image_size
        ), f"图片尺寸异常：{data_path} 中的 image 形状为 {image.shape}"

        return image, data, data_file

    def __getitem__(self, idx):
        # 取出 image, data, data_file
        image, data, data_file = self._get_data(idx)

        # 其他字段
        target = data["target"]
        meta_target = data["meta_target"]
        structure_encoded = data["relation_structure_encoded"]
        del data  # 释放内存

        # 如果需要做额外 transform，则把 numpy 转为 Tensor 交给 transform
        if self.transform:
            # transform 期望的是一个 FloatTensor 格式，例如 (16,H,W)
            image = torch.from_numpy(image).type(torch.float32)
            image = self.transform(image)

        # 转成 torch.Tensor
        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(meta_target, dtype=torch.float32)
        structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)

        # 如果训练阶段不需要 data_file，可以直接 return (image, target, meta_target, structure_encoded)
        return image, target, meta_target, structure_encoded, data_file




