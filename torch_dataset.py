import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

data_transformer = transforms.Compose([transforms.ToTensor()])

torch.manual_seed(0)  # fix random seed


class pytorch_data(Dataset):

    def __init__(self, data_dir, transform, data_type="train"):
        file_path = data_dir + '/' + data_type + "_data.txt"
        self.full_filenames = []
        self.labels = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            words = line.split()  # 将每一行按空格分割成单词
            self.full_filenames.append(words[0])
            self.labels.append(int(words[1]))
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)  # size of dataset

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Open Image with PIL
        image = self.transform(image)  # Apply Specific Transformation to Image
        return image, self.labels[idx]


def get_dataset(data_dir=r'./dataset/', params_model=None):
    tr_transf = transforms.Compose([
        transforms.Resize((params_model['shape_in'][1], params_model['shape_in'][2])),
        transforms.ToTensor()])

    # For the validation dataset, we don't need any augmentation; simply convert images into tensors
    ts_transf = transforms.Compose([
        transforms.Resize((params_model['shape_in'][1], params_model['shape_in'][2])),
        transforms.ToTensor()])
    train_ts = pytorch_data(data_dir, tr_transf, "train")
    test_ts = pytorch_data(data_dir, ts_transf, "test")

    print("train dataset size:", len(train_ts))
    print("validation dataset size:", len(test_ts))
    return test_ts, test_ts
