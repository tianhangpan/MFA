from pathlib import Path
import argparse
import threading
import multiprocessing
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import re
import cv2


class MultiProcessor:
    def __init__(self):
        self.args = self.parse_args()
        self.root = Path(self.args.root)
        self.target = Path(self.args.target)
        self.multiplying_power = self.args.multiplying_power
        self.cpu_count = multiprocessing.cpu_count()
        self.fix = None if self.args.fix == 'None' else float(self.args.fix)

        self.target.mkdir(exist_ok=False)
        self.phase_list = ['train', 'test']
        self.task_list = []

        for phase in ['train', 'test']:
            video_ls = {'train': [3, 4, 5, 6], 'test': [0, 1, 2, 7, 8, 9]}[phase]
            for video_number in video_ls:
                video_path = self.root / 'vidf' / f'vidf1_33_00{video_number}.y'
                for img_path in video_path.glob('*.png'):
                    img_name = img_path.name
                    self.task_list.append([phase, video_number, img_name])

        # print(self.task_list)
        # print(len(self.task_list))

        self.threads = [threading.Thread(target=self.task, args=(i,)) for i in range(self.cpu_count)]
        self.present_index = 0
        self.done_number = 0
        self.lock = threading.RLock()
        print(f'count of threads: {self.cpu_count}')

    def task(self, processor_id):
        while True:
            self.lock.acquire()
            if self.present_index >= len(self.task_list):
                self.lock.release()
                return
            task = self.task_list[self.present_index]
            self.present_index += 1
            self.lock.release()

            img_path = self.root / 'vidf' / f'vidf1_33_00{task[1]}.y' / task[2]
            mat_path = self.root / 'vidf-cvpr' / f'vidf1_33_00{task[1]}_frame_full.mat'

            tar_img_path = self.target / task[0] / str(task[1]) / 'images'
            tar_gt_path = self.target / task[0] / str(task[1]) / 'ground_truth'
            tar_amb_gt_path = self.target / task[0] / str(task[1]) / 'amb_gt'

            for path in [tar_img_path, tar_gt_path, tar_amb_gt_path]:
                path.mkdir(exist_ok=True, parents=True)

            img_num_s = re.findall(r'^vidf1_33_\d+_f(\d+)\.\w+$', task[2])[0]
            img_num = int(img_num_s)
            img_save_path = tar_img_path / (img_num_s + '.jpg')
            img = Image.open(img_path)
            img = img.resize((952, 632))
            img.save(img_save_path, quality=95)

            points = scipy.io.loadmat(str(mat_path))['frame'][0][img_num-1][0][0][0][:, :2]
            num_of_people = points.shape[0]

            img = plt.imread(img_path)
            gt_density_map = self.gaussian_filter_density(img, points, None, mode='gt', fix=self.fix)
            amb_density_map = self.gaussian_filter_density(img, points, self.multiplying_power,
                                                           mode='amb', fix=self.fix)

            gt_density_map = cv2.resize(gt_density_map, (952, 632)) * (238 / 952) * (158 / 632)
            amb_density_map = cv2.resize(amb_density_map, (952, 632)) * (238 / 952) * (158 / 632)

            gt_save_path = tar_gt_path / (img_num_s + '.h5')
            h5f = h5py.File(str(gt_save_path), 'w')
            h5f.create_dataset('density', data=gt_density_map)
            h5f.close()

            amb_save_path = tar_amb_gt_path / (img_num_s + '.npy')
            np.save(str(amb_save_path), amb_density_map)

            self.lock.acquire()
            self.done_number += 1
            print(f'Thread {processor_id:<2}: [{self.done_number}/{len(self.task_list)}] '
                  f'{img_path} processed. num people: {num_of_people}.')
            self.lock.release()

    def start(self):
        for thread in self.threads:
            thread.daemon = True
            thread.start()

        for thread in self.threads:
            thread.join()

        roi_mat_path = self.root / 'vidf-cvpr' / 'vidf1_33_roi_mainwalkway.mat'
        roi_mask = scipy.io.loadmat(str(roi_mat_path))['roi'][0][0][2]
        roi_mask = cv2.resize(roi_mask, (952, 632))
        roi_save_path = self.target / 'roi.npy'
        np.save(str(roi_save_path), roi_mask)
        print(f'Roi mask saved to {roi_save_path}')

    # partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
    @staticmethod
    def gaussian_filter_density(img, points, multiplying_power, mode='gt', fix=None):
        """
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        img_shape: the shape of the image, same as the shape of required density-map. (row,col).
        Note that can not have channel.

        return:
        density: the density-map we want. Same shape as input image but only has one channel.

        example:
        points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
        img_shape: (768,1024) 768 is row and 1024 is column.
        """
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
            if fix:
                sigma = fix
            if mode == 'amb':
                sigma *= multiplying_power
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        # print('done.')
        density = np.clip(density, 0., 1.)
        return density

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-r', '--root', help='original data path',
                            default=r'/Users/pantianhang/python_data/datasets/ucsdpeds')
        parser.add_argument('-t', '--target', help='target data path',
                            default=r'/Users/pantianhang/python_data/datasets/ucsd_fix')
        parser.add_argument('-mp', '--multiplying_power', type=float, default=1.5, help='multiplying power of amb mode')
        parser.add_argument('-f', '--fix', type=str, default='5', help='whether to fix the sigma of gaussian kernel')
        args = parser.parse_args()
        return args


if __name__ == '__main__':
    mp = MultiProcessor()
    mp.start()
