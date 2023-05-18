import torch
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
from itertools import chain

from dataset.image import LoadData


class BaseDataset(Dataset):
    def __init__(self, root, transform, mask, diff_calculation, crop_size=None, train=True, csr_mode=False):
        super().__init__()
        self.lines = root
        self.crop_size = crop_size
        self.len_lines = [len(line_ele) for line_ele in self.lines]
        self.front_boundaries = []
        self.back_boundaries = []
        last_back_boundary = -1
        for len_line_ele in self.len_lines:
            self.front_boundaries.append(last_back_boundary + 1)
            self.back_boundaries.append(last_back_boundary + len_line_ele)
            last_back_boundary = self.back_boundaries[-1]
        self.lines = list(chain(*self.lines))
        self.rand_mask = list(range(len(self.lines)))
        self.transform = transform
        self.mask = mask
        self.diff_calculation = diff_calculation
        self.train = train
        self.csr_mode = csr_mode

    def __len__(self):
        return len(self.rand_mask)

    def trans(self, img, amb_target, target, mask, *support_imgs):
        if self.csr_mode:
            target_new_shape = (amb_target.shape[1] // 8, amb_target.shape[0] // 8)
        else:
            target_new_shape = ((amb_target.shape[1] // 32) * 4, (amb_target.shape[0] // 32) * 4)
        rate = (amb_target.shape[1] / target_new_shape[0]) * (amb_target.shape[0] / target_new_shape[1])
        amb_target = cv2.resize(amb_target, target_new_shape) * rate
        target = cv2.resize(target, target_new_shape) * rate

        img = np.array(img)
        diff = []
        support_img_tensors = []
        np_ones = np.ones_like(img).astype('float32')
        if support_imgs[0]:
            for support_img in support_imgs:
                support_img = np.array(support_img)
                diff.append(self.transform(np.abs(np.log(img + np_ones) - np.log(support_img + np_ones))
                                           if self.diff_calculation == 'div'
                                           else np.abs(img - support_img)))
                support_img_tensors.append(self.transform(support_img))
            if len(support_imgs) == 2:
                support_f, support_b = support_imgs
                diff.append(self.transform(np.abs(np.log(support_b + np_ones) - np.log(support_f + np_ones))))
        img = self.transform(img)
        amb_target = torch.tensor(amb_target).type(torch.FloatTensor).unsqueeze(0)
        target = torch.tensor(target).type(torch.FloatTensor).unsqueeze(0)
        if not isinstance(mask, type(None)):
            mask = cv2.resize(mask, target_new_shape)
            mask = torch.tensor(mask).unsqueeze(0)
        else:
            mask = torch.ones_like(target)

        return img, amb_target, target, mask, support_img_tensors, diff

    @staticmethod
    def get_rand_scale():
        return {'x': random.random(), 'y': random.random(), 'flip': random.random()}


class PresentDataset(BaseDataset):
    def __init__(self, root, transform, mask, crop_size=None, train=True, csr_mode=False):
        super(PresentDataset, self).__init__(root, transform, mask, None, crop_size, train, csr_mode)
        if self.train:
            if len(self.rand_mask) < 3000:
                self.rand_mask *= 4
            elif len(self.rand_mask) < 100:
                self.rand_mask *= 50
            random.shuffle(self.rand_mask)

    def __getitem__(self, index):
        mask = self.mask
        if self.train:
            scale = self.get_rand_scale()
            img, amb_target, target, mask = LoadData.train_data(self.lines[self.rand_mask[index]],
                                                                self.crop_size, scale, mask)
        else:
            img, amb_target, target = LoadData.test_data(self.lines[self.rand_mask[index]])
        img, amb_target, target, mask, _, _ = self.trans(img, amb_target, target, mask, None)

        return img, amb_target, target, mask


class FrontBackDataset(BaseDataset):
    def __init__(self, root, transform, mask, diff_calculation, crop_size=None, train=True, csr_mode=False):
        super().__init__(root, transform, mask, diff_calculation, crop_size, train, csr_mode)
        self.rand_mask = list(set(self.rand_mask) - set(self.front_boundaries) - set(self.back_boundaries))
        if self.train:
            if len(self.rand_mask) < 100:
                self.rand_mask *= 50
            elif len(self.rand_mask) < 3000:
                self.rand_mask *= 4
            random.shuffle(self.rand_mask)

    def __getitem__(self, index):
        mask = self.mask
        if self.train:
            scale = self.get_rand_scale()
            img, amb_target, target, mask = LoadData.train_data(self.lines[self.rand_mask[index]],
                                                                self.crop_size, scale, mask)
            support_f = LoadData.train_data(self.lines[self.rand_mask[index]-1],
                                            self.crop_size, scale, mask, only_img=True)
            support_b = LoadData.train_data(self.lines[self.rand_mask[index]+1],
                                            self.crop_size, scale, mask, only_img=True)
        else:
            img, amb_target, target = LoadData.test_data(self.lines[self.rand_mask[index]])
            support_f = LoadData.test_data(self.lines[self.rand_mask[index]-1], only_img=True)
            support_b = LoadData.test_data(self.lines[self.rand_mask[index]+1], only_img=True)
        img, amb_target, target, mask, [support_f, support_b], [diff_f, diff_b, diff_fb] = \
            self.trans(img, amb_target, target, mask, support_f, support_b)
        return img, amb_target, target, mask, support_f, support_b, diff_f, diff_b, diff_fb


# class FrontDataset(BaseDataset):
#     def __init__(self, root, transform, mask, diff_calculation, train=True):
#         super().__init__(root, transform, mask, diff_calculation, train)
#         self.rand_mask = list(set(self.rand_mask) - set(self.front_boundaries))
#         if self.train:
#             self.rand_mask *= 4
#             random.shuffle(self.rand_mask)
#
#     def __getitem__(self, index):
#         mask = self.mask.copy()
#         if self.train:
#             scale = self.get_rand_scale()
#             img, amb_target, target, mask = LoadData.train_data(self.lines[self.rand_mask[index]], scale, mask)
#             support_img, _, _, _ = LoadData.train_data(self.lines[self.rand_mask[index]-1], scale, mask)
#         else:
#             img, amb_target, target = LoadData.test_data(self.lines[self.rand_mask[index]])
#             support_img, _, _ = LoadData.test_data(self.lines[self.rand_mask[index]-1])
#
#         img, amb_target, target, mask, [support_img], [diff] = \
#             self.trans(img, amb_target, target, mask, support_img)
#         return img, amb_target, target, mask, support_img, diff
#
#
# class BackDataset(BaseDataset):
#     def __init__(self, root, transform, mask, diff_calculation, train=True):
#         super().__init__(root, transform, mask, diff_calculation, train)
#         self.rand_mask = list(set(self.rand_mask) - set(self.back_boundaries))
#         if self.train:
#             self.rand_mask *= 4
#             random.shuffle(self.rand_mask)
#
#     def __getitem__(self, index):
#         mask = self.mask
#         if self.train:
#             scale = self.get_rand_scale()
#             img, amb_target, target, mask = LoadData.train_data(self.lines[self.rand_mask[index]], scale, mask)
#             support_img, _, _, _ = LoadData.train_data(self.lines[self.rand_mask[index]+1], scale, mask)
#         else:
#             img, amb_target, target = LoadData.test_data(self.lines[self.rand_mask[index]])
#             support_img, _, _ = LoadData.test_data(self.lines[self.rand_mask[index]+1])
#
#         img, amb_target, target, mask, [support_img], [diff] = \
#             self.trans(img, amb_target, target, mask, support_img)
#         return img, amb_target, target, mask, support_img, diff
