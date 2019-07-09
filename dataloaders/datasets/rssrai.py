from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from dataloaders import custom_transforms as tr
import albumentations as A
from mypath import Path
from tqdm import tqdm


class Rssrai(data.Dataset):
    NUM_CLASSES = 16

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('rssrai'),
                 split='train',
                 ):
        """
        :param base_dir: path to rssrai dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'img_npy')
        self._cat_dir = os.path.join(self._base_dir, 'label_npy')
        self._mean_path = os.path.join(self._base_dir, "mean.npy")
        self._std_path = os.path.join(self._base_dir, "std.npy")
        self.split = split
        self.args = args
        self.mean = None
        self.std = None
        self.test_size = 0.1

        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.categories = []

        # 加载数据
        train_csv_url = os.path.join(self._base_dir, 'rssrai.csv')
        data = pd.read_csv(train_csv_url)
        tr, vd = train_test_split(data, test_size=self.test_size, random_state=123)
        df = None
        if "train" == self.split:
            df = tr
        elif "val" == self.split:
            df = vd
        # 切分训练集和验证集
        for row in df.itertuples(index=True, name='Pandas'):
            _image = os.path.join(self._image_dir, getattr(row, "img"))
            _category = os.path.join(self._cat_dir, getattr(row, "label"))
            assert os.path.isfile(_image)
            assert os.path.isfile(_category)
            self.im_ids.append(getattr(row, "index"))
            self.images.append(_image)
            self.categories.append(_category)
        assert (len(self.images) == len(self.categories))

        self._get_mean_std()

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))

    def _get_mean_std(self):
        if not os.path.exists(self._mean_path) \
                and \
                not os.path.exists(self._std_path):
            print("compute mean and std value")
            images_npy = np.load(self.images[0]).reshape(1, 256, 256, 4)
            for image_npy_path in tqdm(self.images[1:]):
                images_npy = np.concatenate((images_npy, np.load(image_npy_path).reshape(1, 256, 256, 4)), axis=0)
            print(images_npy.shape)
            images_npy = images_npy / 255
            self.mean = np.mean(images_npy, axis=(0, 1, 2))
            self.std = np.std(images_npy, axis=(0, 1, 2))
            np.save(self._mean_path, self.mean)
            np.save(self._std_path, self.std )
        else:
            print("loading mean and std value")
            self.mean = np.load(self._mean_path)
            self.std  = np.load(self._std_path)
        print(f"mean: {self.mean}")
        print(f"std: {self.std}")

    def __getitem__(self, index):
        _img, _target = self._read_numpy_file(index)
        sample = {'image': _img, 'label': _target}
        return self.transform(sample)

    def __len__(self):
        return len(self.images)

    def _read_numpy_file(self, index):
        _img = np.load(self.images[index]).astype('float64')
        _target = np.load(self.categories[index]).astype('float64')

        return _img, _target

    def transform(self, sample):
        image = A.Normalize(mean=self.mean, std=self.std)(image=sample['image'])["image"]
        image, label = torch.from_numpy(image), torch.from_numpy(sample['label'])
        sample['image'], sample['label'] = image.permute(2, 0, 1), label
        return sample

    def __str__(self):
        return 'Rssrai(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    rssrai_val = Rssrai(args, split='val')

    dataloader = DataLoader(rssrai_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        sample['image'] = sample['image'][:, 1:, :, :]
        for jj in range(sample["image"].size()[0]):
            # print(f"img:{sample['image'].size()},label:{sample['label'].size()}")
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            # img_tmp *= (0.229, 0.224, 0.225)
            # img_tmp += (0.485, 0.456, 0.406)
            # img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
