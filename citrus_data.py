import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

# import cv2
import random


# CitrusDisease6数据集处理
class CitrusDisease6_preprocess():
    def __init__(self, path=None):
        self.raw_path = 'E:/Data/yu_data/org_data_pre1/'
        self.path = path
        self.label_names = ['anthracnose', 'canker', 'huanglong', 'melanose', 'normal', 'sunscald']
        # self.label_names = {'0': 'normal', '1': 'anthracnose', '2': 'canker',
        #                     '3': 'huanglong', '4': 'melanose', '5': 'sunscald'}
        self.class_num = 6
        self.data_splits = ['_train', '_test', '_pretrain', '_pretest']
        self.image_size = (512, 512)

    def image_rename(self):
        folders = os.listdir(self.raw_path)

        for i in range(len(folders)):
            folder_path = os.path.join(self.raw_path, self.label_names[str(i)])
            save_path = os.path.join(self.path, self.label_names[str(i)])
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            image_names = os.listdir(folder_path)
            image_names = self.remove_non_image_files(image_names)

            for j in range(len(image_names)):
                image = Image.open(os.path.join(folder_path, image_names[j]))
                resize_image = image.resize(self.image_size)
                rename = str(i) + '_' + self.label_names[str(i)] + '_' + str(j) + '.jpg'

                resize_image.save(os.path.join(save_path, rename))
                print(f"Image {rename} is saved successfully!")

    def remove_non_image_files(self, image_names):
        new_image_names = []
        for i in range(len(image_names)):
            if image_names[i].split('.')[-1] == 'jpg' or image_names[i].split('.')[-1] == 'JPG':
                new_image_names.append(image_names[i])
        return new_image_names

    def train_test_split(self, split_rate=0.95):
        train_dic, test_dic = {}, {}
        train_data, train_labels, train_filenames = [], [], []
        test_data, test_labels, test_filenames = [], [], []
        for i in range(self.class_num):
            folder_path = os.path.join(self.path, self.label_names[i])

            image_names = os.listdir(folder_path)
            # image_names = self.remove_non_image_files(image_names)

            for j in range(len(image_names)):
                # with open(os.path.join(folder_path, image_names[j].split('.')[0]), "rb") as f:
                #     image = pickle.load(f, encoding="latin1")
                # image = np.load(os.path.join(folder_path, image_names[j].split('.')[0] + '.npy'), encoding='bytes', allow_pickle=True)

                # image = cv2.imread(os.path.join(folder_path, image_names[j]))
                image = Image.open(os.path.join(folder_path, image_names[j]))
                image = np.array(image)

                if j < int(len(image_names) * split_rate):
                # if int(len(image_names) * 0.5) <= j < int(len(image_names) * split_rate):
                    train_data.append(image)
                    train_labels.append(int(image_names[j].split('_')[0]))
                    train_filenames.append(image_names[j])
                    print(f"Image {image_names[j]} is added in train set!")
                elif int(len(image_names) * split_rate) <= j:
                    test_data.append(image)
                    test_labels.append(int(image_names[j].split('_')[0]))
                    test_filenames.append(image_names[j])
                    print(f"Image {image_names[j]} is added in test set!")

        train_data_labels = list(zip(train_data, train_labels, train_filenames))
        random.shuffle(train_data_labels)
        train_data, train_labels, train_filenames = zip(*train_data_labels)
        test_data_labels = list(zip(test_data, test_labels, test_filenames))
        random.shuffle(test_data_labels)
        test_data, test_labels, test_filenames = zip(*test_data_labels)

        train_data = np.array(train_data)
        test_data = np.array(test_data)

        train_dic['data'] = train_data
        train_dic['labels'] = train_labels
        train_dic['filenames'] = train_filenames
        train_dic['class_num'] = self.class_num
        train_dic['label_names'] = self.label_names
        test_dic['data'] = test_data
        test_dic['labels'] = test_labels
        test_dic['filenames'] = test_filenames
        test_dic['class_num'] = self.class_num
        test_dic['label_names'] = self.label_names

        train_file = os.path.join(self.path, self.data_splits[0])
        output = open(train_file, 'wb')
        pickle.dump(train_dic, output)
        output.close()

        test_file = os.path.join(self.path, self.data_splits[1])
        output = open(test_file, 'wb')
        pickle.dump(test_dic, output)
        output.close()

    def pre_train_test_split(self, split_rate=(0.8, 0.95)):
        pretrain_dic, train_dic, test_dic = {}, {}, {}
        pretrain_data, pretrain_labels, pretrain_filenames = [], [], []
        train_data, train_labels, train_filenames = [], [], []
        test_data, test_labels, test_filenames = [], [], []
        for i in range(self.class_num):
            folder_path = os.path.join(self.path, self.label_names[i])

            image_names = os.listdir(folder_path)
            # image_names = self.remove_non_image_files(image_names)

            for j in range(len(image_names)):
                # with open(os.path.join(folder_path, image_names[j].split('.')[0]), "rb") as f:
                #     image = pickle.load(f, encoding="latin1")
                # image = np.load(os.path.join(folder_path, image_names[j].split('.')[0] + '.npy'), encoding='bytes', allow_pickle=True)

                # image = cv2.imread(os.path.join(folder_path, image_names[j]))
                image = Image.open(os.path.join(folder_path, image_names[j]))
                image = np.array(image)

                if j < int(len(image_names) * split_rate[0]):
                    pretrain_data.append(image)
                    pretrain_labels.append(int(image_names[j].split('_')[0]))
                    pretrain_filenames.append(image_names[j])
                    print(f"Image {image_names[j]} is added in pretrain set!")
                elif int(len(image_names) * split_rate[0]) <= j < int(len(image_names) * split_rate[1]):
                    train_data.append(image)
                    train_labels.append(int(image_names[j].split('_')[0]))
                    train_filenames.append(image_names[j])
                    print(f"Image {image_names[j]} is added in train set!")
                elif int(len(image_names) * split_rate[1]) <= j:
                    test_data.append(image)
                    test_labels.append(int(image_names[j].split('_')[0]))
                    test_filenames.append(image_names[j])
                    print(f"Image {image_names[j]} is added in test set!")

        # pretrain_data_labels = list(zip(pretrain_data, pretrain_labels, pretrain_filenames))
        # random.shuffle(pretrain_data_labels)
        # pretrain_data, pretrain_labels, pretrain_filenames = zip(*pretrain_data_labels)
        # train_data_labels = list(zip(train_data, train_labels, train_filenames))
        # random.shuffle(train_data_labels)
        # train_data, train_labels, train_filenames = zip(*train_data_labels)
        # test_data_labels = list(zip(test_data, test_labels, test_filenames))
        # random.shuffle(test_data_labels)
        # test_data, test_labels, test_filenames = zip(*test_data_labels)

        pretrain_data = np.array(pretrain_data)
        train_data = np.array(train_data)
        test_data = np.array(test_data)

        pretrain_dic['data'] = pretrain_data
        pretrain_dic['labels'] = pretrain_labels
        pretrain_dic['filenames'] = pretrain_filenames
        pretrain_dic['class_num'] = self.class_num
        pretrain_dic['label_names'] = self.label_names
        train_dic['data'] = train_data
        train_dic['labels'] = train_labels
        train_dic['filenames'] = train_filenames
        train_dic['class_num'] = self.class_num
        train_dic['label_names'] = self.label_names
        test_dic['data'] = test_data
        test_dic['labels'] = test_labels
        test_dic['filenames'] = test_filenames
        test_dic['class_num'] = self.class_num
        test_dic['label_names'] = self.label_names

        pretrain_file = os.path.join(self.path, self.data_splits[2])
        output = open(pretrain_file, 'wb')
        pickle.dump(pretrain_dic, output)
        output.close()

        train_file = os.path.join(self.path, self.data_splits[0])
        output = open(train_file, 'wb')
        pickle.dump(train_dic, output)
        output.close()

        test_file = os.path.join(self.path, self.data_splits[1])
        output = open(test_file, 'wb')
        pickle.dump(test_dic, output)
        output.close()

    def pretrain_pretest_split(self, split_rate=0.8):
        pretrain_dic, pretest_dic = {}, {}
        pretrain_data, pretrain_labels, pretrain_filenames = [], [], []
        pretest_data, pretest_labels, pretest_filenames = [], [], []
        for i in range(self.class_num):
            folder_path = os.path.join(self.path, self.label_names[i])

            image_names = os.listdir(folder_path)
            # image_names = self.remove_non_image_files(image_names)

            for j in range(len(image_names)):
                # with open(os.path.join(folder_path, image_names[j].split('.')[0]), "rb") as f:
                #     image = pickle.load(f, encoding="latin1")
                # image = np.load(os.path.join(folder_path, image_names[j].split('.')[0] + '.npy'), encoding='bytes', allow_pickle=True)

                # image = cv2.imread(os.path.join(folder_path, image_names[j]))
                image = Image.open(os.path.join(folder_path, image_names[j]))
                image = np.array(image)

                if j < int(len(image_names) * split_rate):
                    pretrain_data.append(image)
                    pretrain_labels.append(int(image_names[j].split('_')[0]))
                    pretrain_filenames.append(image_names[j])
                    print(f"Image {image_names[j]} is added in pretrain set!")
                elif int(len(image_names) * split_rate) <= j:
                    pretest_data.append(image)
                    pretest_labels.append(int(image_names[j].split('_')[0]))
                    pretest_filenames.append(image_names[j])
                    print(f"Image {image_names[j]} is added in test set!")

        # pretrain_data_labels = list(zip(pretrain_data, pretrain_labels, pretrain_filenames))
        # random.shuffle(pretrain_data_labels)
        # pretrain_data, pretrain_labels, pretrain_filenames = zip(*pretrain_data_labels)
        # train_data_labels = list(zip(train_data, train_labels, train_filenames))
        # random.shuffle(train_data_labels)
        # train_data, train_labels, train_filenames = zip(*train_data_labels)
        # test_data_labels = list(zip(test_data, test_labels, test_filenames))
        # random.shuffle(test_data_labels)
        # test_data, test_labels, test_filenames = zip(*test_data_labels)

        pretrain_data = np.array(pretrain_data)
        pretest_data = np.array(pretest_data)

        pretrain_dic['data'] = pretrain_data
        pretrain_dic['labels'] = pretrain_labels
        pretrain_dic['filenames'] = pretrain_filenames
        pretrain_dic['class_num'] = self.class_num
        pretrain_dic['label_names'] = self.label_names
        pretest_dic['data'] = pretest_data
        pretest_dic['labels'] = pretest_labels
        pretest_dic['filenames'] = pretest_filenames
        pretest_dic['class_num'] = self.class_num
        pretest_dic['label_names'] = self.label_names

        # pretrain_file = os.path.join(self.path, self.data_splits[2])
        # output = open(pretrain_file, 'wb')
        # pickle.dump(pretrain_dic, output)
        # output.close()

        pretest_file = os.path.join(self.path, self.data_splits[3])
        output = open(pretest_file, 'wb')
        pickle.dump(pretest_dic, output)
        output.close()


class CitrusDisease6(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "citrus_disease_6"
    train_file = "train"
    test_file = "test"
    pretrain_file = 'pretrain'
    pretest_file = 'pretest'

    def __init__(
        self,
        root: str,
        pretrain: bool = True,
        train: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.pretrain = pretrain
        self.train = train

        if self.pretrain:
            data = self.pretrain_file
        elif self.train:
            data = self.train_file
        else:
            data = self.test_file

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        file_path = os.path.join(self.root, self.base_folder, data)
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            self.data = entry["data"]
            self.targets.extend(entry["labels"])
            self.classes = entry["label_names"]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            # img = self.transform(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


# class CitrusDisease6(VisionDataset):
#     """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
#
#     Args:
#         root (string): Root directory of dataset where directory
#             ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
#         train (bool, optional): If True, creates dataset from training set, otherwise
#             creates from test set.
#         transform (callable, optional): A function/transform that takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#
#     """
#
#     base_folder = "citrus-disease-6"
#     train_file = "train"
#     test_file = "test"
#     pretrain = 'pretrain'
#
#     def __init__(
#         self,
#         root: str,
#         pretrain: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#     ) -> None:
#
#         super().__init__(root, transform=transform, target_transform=target_transform)
#
#         self.train = train  # training set or test set
#
#         if self.train:
#             data = self.train_file
#         else:
#             data = self.test_file
#
#         self.data: Any = []
#         self.targets = []
#
#         # now load the picked numpy arrays
#         file_path = os.path.join(self.root, self.base_folder, data)
#         with open(file_path, "rb") as f:
#             entry = pickle.load(f, encoding="latin1")
#             self.data = entry["data"]
#             self.targets.extend(entry["labels"])
#             self.classes = entry["label_names"]
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.targets[index]
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             # img = self.transform(img)
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def extra_repr(self) -> str:
#         split = "Train" if self.train is True else "Test"
#         return f"Split: {split}"



if __name__ == "__main__":
    P = CitrusDisease6_preprocess(path=os.getcwd() + '/datasets/citrus-disease-6/')
    # P.image_rename()
    # P.train_test_split()
    # P.pre_train_test_split()
    P.pretrain_pretest_split()