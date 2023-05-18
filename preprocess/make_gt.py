import argparse

import h5py
import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
import os
from matplotlib import pyplot as plt
from pathlib import Path
import threading
import multiprocessing
import json
import re


class MultiProcessor:
    def __init__(self):
        self.args = self.parse_args()
        self.mode = self.args.mode
        self.root = Path(self.args.origin_dir)
        self.cpu_count = multiprocessing.cpu_count()
        self.multiplying_power = self.args.multiplying_power

        self.number = self.args.number
        self.weight_path = self.args.weight_path
        self.weight_map = io.loadmat(self.weight_path)['weightImgAll'][0][self.number] if self.number != -1 else None

        self.train_path = self.root / 'train' / 'images'
        self.test_path = self.root / 'test' / 'images'
        self.path_sets = [self.train_path, self.test_path]
        self.img_paths = [[], []]
        for phase, path in enumerate(self.path_sets):
            for img_path in path.glob('*.jpg'):
                self.img_paths[phase].append(str(img_path))
        self.present_phase = 0
        self.present_index = 0
        self.phase_list = ['train', 'test']
        self.save_dirs = [self.root / phase / 'amb_gt' for phase in self.phase_list]
        for save_dir in self.save_dirs:
            if not os.path.exists(str(save_dir)):
                os.makedirs(str(save_dir))

        self.log_path = self.root / 'log.json'
        if not self.log_path.exists():
            jpg_paths = self.train_path.glob('*.jpg')
            img_stem_example = jpg_paths.__next__().stem
            image_prefix = re.findall(r'(\w+)_\d+$', img_stem_example)[0] + '_'
            log_dict = {'multiplying_power': 1.5, 'image_prefix': image_prefix}
            with self.log_path.open(mode='w') as jsf:
                log_str = json.dumps(log_dict)
                jsf.write(log_str)
            print('json file created.')

        with self.log_path.open(mode='r') as jsf:
            log_dict = json.load(jsf)
            self.mp_match = (log_dict['multiplying_power'] == self.multiplying_power) if self.mode == 'amb' else None
            self.image_prefix = log_dict['image_prefix']

        self.threads = [threading.Thread(target=self.task, args=(i,)) for i in range(self.cpu_count)]
        print(f'count of threads: {self.cpu_count}')

    def task(self, processor_id):
        while True:
            if self.present_phase >= 2:
                return
            else:
                if self.present_index >= len(self.img_paths[self.present_phase]):
                    self.present_phase += 1
                    self.present_index = 0
                    continue
                else:
                    phase = self.present_phase
                    index = self.present_index
                    self.present_index += 1
                    list_len = len(self.img_paths[phase])
                    img_path = self.img_paths[phase][index]
                    name = os.path.basename(img_path)
                    print(f'thread {processor_id:<2}:phase[{phase+1}/2] image[{index+1}/{list_len}] '
                          f'processing {img_path}')

                    # mat = io.loadmat(
                    #     img_path.replace('.jpg', '.mat').replace('images', 'ground_truth')
                    #         .replace(self.image_prefix, 'GT_IMG_'))
                    # points = mat["image_info"][0][0][0][0][1]  # 1546person*2(col,row)

                    points = np.load(img_path.replace('.jpg', '.npy').replace('images', 'ground_truth')
                                     .replace(self.image_prefix, 'GT_IMG_'))

                    img = plt.imread(img_path)
                    # k = np.zeros((img.shape[0], img.shape[1]))

                    k = self.gaussian_filter_density(img, points, self.multiplying_power, mode=self.mode)\
                        if self.number == -1 else \
                        self.weighted_density(img, points, self.weight_map, self.multiplying_power, mode=self.mode)
                    # save density_map to disk
                    im_save_path = str(self.save_dirs[phase] / name)
                    gd_save_path = im_save_path.replace('jpg', 'npy')
                    if self.mode == 'amb':
                        # print('gd save path:', gd_save_path)
                        np.save(gd_save_path, k)
                        print(f'thread {processor_id:<2}: {gd_save_path} saved')
                    else:
                        h5_path = gd_save_path.replace('amb_gt', 'ground_truth').replace('.npy', '.h5')
                        h5f = h5py.File(h5_path, 'w')
                        h5f.create_dataset('density', data=k)
                        h5f.close()
                        print(f'thread {processor_id:<2}: {h5_path} saved')

    def start(self):
        if self.mode == 'amb':
            if self.mp_match:
                print('multiplying power matched.')
                return
            else:
                with self.log_path.open(mode='r') as jsf:
                    log_dict = json.load(jsf)
                with self.log_path.open(mode='w') as jsf:
                    log_dict['multiplying_power'] = 'changing'
                    log_str = json.dumps(log_dict)
                    jsf.write(log_str)

        for thread in self.threads:
            thread.setDaemon(True)
            thread.start()

        for thread in self.threads:
            thread.join()

        if self.mode == 'amb':
            with self.log_path.open(mode='r') as jsf:
                log_dict = json.load(jsf)
            with self.log_path.open(mode='w') as jsf:
                log_dict['multiplying_power'] = self.multiplying_power
                log_str = json.dumps(log_dict)
                jsf.write(log_str)

        print('done.')

    # partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
    @staticmethod
    def gaussian_filter_density(img, points, multiplying_power, mode='gt'):
        '''
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

        return:
        density: the density-map we want. Same shape as input image but only has one channel.

        example:
        points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
        img_shape: (768,1024) 768 is row and 1024 is column.
        '''
        img_shape = [img.shape[0], img.shape[1]]
        # print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
        density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)
        if gt_count == 0:
            return density

        leafsize = 2048
        # build kdtree
        tree = KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(points, k=4)

        # print('generate density...')
        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                pt2d[int(pt[1]), int(pt[0])] = 1.
            else:
                continue
            if gt_count > 1:
                sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            else:
                sigma = np.average(np.array(pt.shape)) / 2. / 2.  # case: 1 point
            sigma = min(20., sigma)
            if mode == 'amb':
                sigma *= multiplying_power
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        # print('done.')
        density = np.clip(density, 0., 1.)
        return density

    @staticmethod
    def weighted_density(img, points, weight_map, multiplying_power, mode='gt'):
        img_shape = [img.shape[0], img.shape[1]]
        density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)
        if gt_count == 0:
            return density

        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                pt2d[int(pt[1]), int(pt[0])] = 1.
            else:
                continue
            sigma = (1. / weight_map[pt[1]][pt[0]]) * 25
            if mode == 'amb':
                sigma *= multiplying_power
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

        return density

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Test ')
        parser.add_argument('origin_dir', default='/home/miaogen_pan/python_data/Shanghai/part_A_final',
                            help='original data directory')
        parser.add_argument('-m', '--mode', type=str, default='gt', help='mode, amb or gt')
        parser.add_argument('-mp', '--multiplying_power', type=float, default=1.5, help='multiplying power of amb mode')
        parser.add_argument('-wp', '--weight_path', type=str, default='None', help='path of weight map')
        parser.add_argument('-n', '--number', type=int, default=-1, help='the number of weight map')
        args = parser.parse_args()
        return args


if __name__ == "__main__":
    multi_processor = MultiProcessor()
    multi_processor.start()
