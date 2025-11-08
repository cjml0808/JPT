# encoding: utf-8
# Copyright (c) 2020-present.
# All rights reserved.
#
# Date: 2021-07-14
# Author: Jiabao Wang
# Email: jiabao_1108@163.com
#

import os
import os.path as osp
from ..utils.data.base_dataset import BaseImageDataset
import pickle
from PIL import Image
import numpy as np


class TomatoDisease5(BaseImageDataset):
    dataset_dir = 'tomatodisease5'

    def __init__(self, root, verbose=True, **kwargs):
        super(TomatoDisease5, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train, test = self._process_dir(self.dataset_dir)
        if verbose:
            print("=> TomatoDisease5 loaded")
            self.num_train_pids, self.num_train_imgs, self.num_test_pids, self.num_test_imgs = \
                self.print_dataset_statistics(train, test)

        self.train = train["data"]
        self.test = test["data"]
        self.train_labels = list(train["labels"])
        self.test_labels = list(test["labels"])
        self.train_filenames = list(train["filenames"])
        self.test_filenames = list(test["filenames"])
        self.class_num = train["class_num"]
        self.label_names = train["label_names"]


    def _process_dir(self, root):
        # now load the picked numpy arrays
        file_path = os.path.join(root, "pretrain")
        with open(file_path, "rb") as f:
            train_dataset = pickle.load(f, encoding="latin1")

        file_path = os.path.join(root, "train")
        with open(file_path, "rb") as f:
            test_dataset = pickle.load(f, encoding="latin1")

        return train_dataset, test_dataset



if __name__ == '__main__':
    dataset = TomatoDisease5()
    for idx, x in enumerate(dataset.train):
        if idx % 500 == 0:
            print(x)
    print("--------------------------------")
    for idx, x in enumerate(dataset.test):
        if idx % 500 == 0:
            print(x)