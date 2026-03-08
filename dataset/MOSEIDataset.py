import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch


class MOSEI_dataset(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(MOSEI_dataset, self).__init__()
        # 兼容 Inforeg 的路径结构，假设数据在 Inforeg/data/MOSEI 下
        dataset_path = os.path.join(dataset_path, data + '_data.pkl' if if_align else data + '_data_noalign.pkl')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        dataset = pickle.load(open(dataset_path, 'rb'))

        # 提取特征
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        self.data = data
        self.n_modalities = 3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        Y = self.labels[index]

        # 将连续值标签转换为三分类（和 mosei 项目一致）
        threshold = 0.5
        if -3 <= Y <= -threshold:
            target = 0
        elif -threshold < Y < threshold:
            target = 1
        else:
            target = 2

        # 返回格式尽量向 Inforeg 靠拢
        # Inforeg AVDataset 返回: spectrogram, image_n, label, av_file
        # 这里返回: text, audio, vision, target
        return self.text[index], self.audio[index], self.vision[index], target


def get_mosei_dataloaders(args, batch_size=32, num_workers=4):
    """
    构建 Train/Val/Test 的 DataLoader
    """
    print("\n构建数据加载器 (Constructing DataLoaders)...")

    # 这里的 args 应该包含 data_path 和 dataset_name
    # 默认 if_align=False，因为 main.py 也没传这个参数，通常使用非对齐数据或者预处理好的数据

    # 1. 实例化训练集
    train_dataset = MOSEI_dataset(
        dataset_path=args.data_path,
        data=args.dataset_name,
        split_type='train',
        if_align=False
    )

    # 2. 实例化验证集
    # 注意：MOSEI 数据集中验证集有时叫 'valid' 有时叫 'dev'，上面 __init__ 里做了兼容处理
    valid_dataset = MOSEI_dataset(
        dataset_path=args.data_path,
        data=args.dataset_name,
        split_type='valid',
        if_align=False
    )

    # 3. 实例化测试集
    test_dataset = MOSEI_dataset(
        dataset_path=args.data_path,
        data=args.dataset_name,
        split_type='test',
        if_align=False
    )

    # 4. 封装 DataLoader
    # Windows 系统下 num_workers 如果大于 0 可能会报错，如果报错请改为 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱
        num_workers=num_workers
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
