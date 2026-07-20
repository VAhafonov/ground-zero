import os

import cv2
import numpy as np
import struct
from array import array
import torch
from torch.utils.data import Dataset
from torchvision import transforms

TRAIN_IMAGES_FILENAME = 'train-images.idx3-ubyte'
TEST_IMAGES_FILENAME = 't10k-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = 'train-labels.idx1-ubyte'
TEST_LABELS_FILENAME = 't10k-labels.idx1-ubyte'

class MnistDataset(Dataset):
    def __init__(self, root_dir: str, is_train: bool):
        self.images = []
        self.labels = []
        self.images_filepath = os.path.join(root_dir, TRAIN_IMAGES_FILENAME if is_train else TEST_IMAGES_FILENAME)
        self.labels_filepath = os.path.join(root_dir, TRAIN_LABELS_FILENAME if is_train else TEST_LABELS_FILENAME)
        self.load()
        self.augs = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),  # MNIST mean and std
            transforms.Resize((224, 224))]
        )

    def __len__(self):
        return len(self.images)

    def load(self):
        self.images, self.labels = self.read_images_labels(self.images_filepath, self.labels_filepath)

    def __getitem__(self, index):
        image = cv2.cvtColor(self.images[index], cv2.COLOR_GRAY2RGB)
        image = self.augs(image)
        label = self.labels[index]
        batch = {'input': image,
                 'label': torch.tensor(label).long()}
        return batch

    @staticmethod
    def read_images_labels(images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i] = img

        return images, labels

def visualise_sample(input_batch, output_batch):
    input_img = input_batch['input'].clone()[0].cpu()
    output = np.argmax(output_batch.clone()[0].cpu().numpy())
    input_img = (input_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    output_img = np.zeros((input_img.shape[0], input_img.shape[1] * 2, 3), dtype=np.uint8)
    output_img[:, :input_img.shape[1], :] = input_img
    cv2.putText(output_img, str(output), (int(input_img.shape[1] * 1.2), int(input_img.shape[0] * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 3)
    return output_img
