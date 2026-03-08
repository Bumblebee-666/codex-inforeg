import csv
import math
import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset
from torchvision import transforms

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def inv_norm_tensor(img):
    inv_norm = UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img[:, :, :] = inv_norm(img)
    return img

class KS_dataset(Dataset): # 147

    def __init__(self, args, mode, select_ratio = 1, v_norm = True, a_norm = False, name = "KS"):
        self.data = []
        self.label = []
        
        if mode=='train':
            csv_path = '/data/Lab105/Datasets/kinetics-dataset/k400/my_train_fixed.txt'
            self.audio_path = '/data/Lab105/Datasets/kinetics-dataset/k400/train_spec'
            self.visual_path = '/data/Lab105/Datasets/kinetics-dataset/k400/train-videos/train-set-img/Image-01-FPS'
        
        elif mode=='val':
            csv_path = '/data/Lab105/Datasets/kinetics-dataset/k400/my_test_fixed.txt'
            self.audio_path = '/data/Lab105/Datasets/kinetics-dataset/k400/test_spec'
            self.visual_path = '/data/Lab105/Datasets/kinetics-dataset/k400/test-videos/test-set-img/Image-01-FPS'

        else:
            csv_path = '/data/Lab105/Datasets/kinetics-dataset/k400/my_test_fixed.txt'
            self.audio_path = '/data/Lab105/Datasets/kinetics-dataset/k400/test_spec'
            self.visual_path = '/data/Lab105/Datasets/kinetics-dataset/k400/test-videos/test-set-img/Image-01-FPS'



        with open(csv_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")
                name = item[0]
                # print(line)

                if os.path.exists(self.audio_path + '/' + name + '.npy'):
                    path = self.visual_path + '/' + name
                    files_list=[lists for lists in os.listdir(path)]
                    if(len(files_list)>3):
                        self.data.append(name)
                        self.label.append(int(item[-1]))

        print('data load finish')
        self.normalize = v_norm
        self.mode = mode
        self._init_atransform()

        if mode=='train' and select_ratio < 1:
            self._random_choice(select_ratio)
        print('# of files = %d ' % len(self.data))

    def _random_choice(self, select_ratio = 0.40):
        num_data = len(self.data)
        selected_id = np.random.choice(np.arange(num_data), int(num_data * select_ratio))
        selected_data = [self.data[idx] for idx in selected_id]
        selected_label = [self.label[idx] for idx in selected_id]
        self.data = selected_data
        self.label = selected_label

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        spectrogram = np.load(self.audio_path + '/' + av_file + '.npy')
        # spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # Visual
        path = self.visual_path + '/' + av_file
        files_list=[lists for lists in os.listdir(path)]
        file_num = len([fn for fn in files_list if fn.endswith("jpg")])

        if self.mode == 'train':
            transf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transf = [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ]
        if self.normalize:
            transf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        transf = transforms.Compose(transf)
        # [新增] 为了崩溃保护，先获取文件夹下所有jpg列表，用于Fallback
        jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

        pick_num = 3
        seg = int(file_num / pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0] * pick_num

        for i in range(pick_num):
            # [保留] 作者原始的采样逻辑 (包括那个奇怪的 >=10 判断)
            if self.mode == 'train':
                t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                if t[i] >= 10:
                    t[i] = 9
            else:
                t[i] = i * seg + max(int(seg / 2), 1) if file_num > 6 else 1

            # === 修改点 1: 文件名匹配问题 ===
            # 作者源码: path1.append('frame_0000' + str(t[i]) + '.jpg')
            # 修改为: 5位数字补零格式 (00001.jpg)，适配你的数据集
            file_name = '{:05d}.jpg'.format(t[i])
            path1.append(file_name)

            full_path = path + "/" + file_name

            # === 修改点 2: 程序崩溃保护 ===
            # 作者源码直接 Image.open，这会导致找不到文件时崩溃
            # 修改为: 先检查是否存在，不存在则随机找一张图代替
            if os.path.exists(full_path):
                img = Image.open(full_path).convert('RGB')
            else:
                # Fallback: 如果计算出的帧不存在，随机选一张存在的
                if len(jpg_files) > 0:
                    fallback = random.choice(jpg_files)
                    img = Image.open(path + "/" + fallback).convert('RGB')
                else:
                    # 极端兜底: 创建纯黑图
                    img = Image.new('RGB', (224, 224))

            image.append(img)

            # [保留] 后续处理保持作者原样
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()

            # [保留] 作者原始的 copy/cat 逻辑
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)

        label = self.label[idx]
        # print(image_n.shape)
        # sample = {
        #     'audio':spectrogram,
        #     'clip':image_n,
        #     'target':label, 
        #     'index' : idx
        # # }
        return spectrogram, image_n, label
