import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def kappa(pre_vec, cor_vec, label_vec):  # testData表示要计算的数据，k表示数据矩阵的是k*k的
        assert len(pre_vec) == len(label_vec) == len(pre_vec)
        tmp = 0.0
        for i in range(len(label_vec)):
            tmp += pre_vec[i] * label_vec[i]
        pe = float(tmp) / sum(label_vec) ** 2
        p0 = float(sum(cor_vec) / sum(label_vec))
        cohens_coefficient = float((p0-pe)/(1-pe))
        return cohens_coefficient           

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Kappa:
    def __init__(self, num_classes=16):
        self.pre_vec = np.zeros(num_classes)
        self.cor_vec = np.zeros(num_classes)
        self.tar_vec = np.zeros(num_classes)
        self.num = num_classes

    def update(self, output, target):
        pre_array = torch.argmax(output, dim=1)
        pre = np.resize(pre_array, (1, -1))
        for i in np.ndarray.flatten(pre):
            self.pre_vec[i] += 1
        
        target = np.resize(target, (1, -1))
        for i in np.ndarray.flatten(target):
            self.tar_vec[i] += 1

        for i in range(self.num):
            pre_mask = (pre_array == i).byte()
            tar_mask = (target == i).byte()
            self.cor_vec[i] = (pre_mask & tar_mask).sum().item()
        
    def get(self):
        assert len(self.pre_vec) == len(self.tar_vec) == len(self.pre_vec)
        tmp = 0.0
        for i in range(len(self.tar_vec)):
            tmp += self.pre_vec[i] * self.tar_vec[i]
        pe = float(tmp) / sum(self.tar_vec) ** 2
        p0 = float(sum(self.cor_vec) / sum(self.tar_vec))
        cohens_coefficient = float((p0-pe)/(1-pe))
        return cohens_coefficient

    def reset(self):
        self.pre_vec = np.zeros(self.num)
        self.cor_vec = np.zeros(self.num)
        self.tar_vec = np.zeros(self.num)