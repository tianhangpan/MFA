from pathlib import Path
import argparse
import threading
import multiprocessing
import json
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
import scipy
import shutil
import matplotlib.pyplot as plt
import h5py
import re
from PIL import Image
import cv2


class MultiProcessor:
    def __init__(self):
        self.args = self.parse_args()
        self.root = Path(self.args.root)
        self.target = Path(self.args.target)
        self.multiplying_power = self.args.multiplying_power
        self.num_workers = self.args.num_workers if self.args.num_workers > 0 else multiprocessing.cpu_count()
        self.resize = True if self.args.resize.lower() == 'true' else False

        self.target.mkdir(exist_ok=True)
        self.phase_list = ['train', 'test']
        self.task_list = []
        for phase in ['train', 'test']:
            phase_path = self.root / phase
            for video_path in phase_path.iterdir():
                video_name = video_path.name
                if video_name[0] == '.':
                    continue
                for img_path in video_path.glob('*.jpg'):
                    if len(re.findall(r'(\d+)\.\w+$', str(img_path))) == 0 or img_path.name[0] == '.':
                        continue
                    img_name = img_path.name
                    self.task_list.append([phase, video_name, img_name])

        self.threads = [threading.Thread(target=self.task, args=(i,)) for i in range(self.num_workers)]
        self.present_index = 0
        self.done_number = 0
        self.lock = threading.RLock()
        print(f'resize: {self.resize}')
        print(f'count of threads: {self.num_workers}')

    def task(self, processor_id):
        while True:
            self.lock.acquire()
            if self.present_index >= len(self.task_list):
                self.lock.release()
                return
            task = self.task_list[self.present_index]
            self.present_index += 1
            self.lock.release()

            img_path = self.root / task[0] / task[1] / task[2]
            json_path = self.root / task[0] / task[1] / task[2].replace('jpg', 'json')

            tar_img_path = self.target / task[0] / task[1] / 'images'
            tar_gt_path = self.target / task[0] / task[1] / 'ground_truth'
            tar_amb_gt_path = self.target / task[0] / task[1] / 'amb_gt'
            for path in [tar_img_path, tar_gt_path, tar_amb_gt_path]:
                path.mkdir(exist_ok=True, parents=True)

            if self.resize:
                img = Image.open(str(img_path))
                img = img.resize((960, 540))
                img_save_path = tar_img_path / task[2]
                img.save(str(img_save_path), quality=95)
            else:
                shutil.copy(img_path, tar_img_path)

            with open(json_path, 'r') as jsf:
                jss = jsf.read()
                state_dict = json.loads(jss)
                key = list(state_dict.keys())[0]
                points = []
                for e in state_dict[key]['regions']:
                    points.append([e['shape_attributes']['x'], e['shape_attributes']['y']])

            img = plt.imread(img_path)
            gt_density_map = self.gaussian_filter_density(img, points, None, mode='gt')
            amb_density_map = self.gaussian_filter_density(img, points, self.multiplying_power, mode='amb')
            raw_h, raw_w = gt_density_map.shape

            if self.resize:
                gt_density_map = cv2.resize(gt_density_map, (960, 540)) * (raw_h / 540) * (raw_w / 960)
                amb_density_map = cv2.resize(amb_density_map, (960, 540)) * (raw_h / 540) * (raw_w / 960)

            gt_save_path = tar_gt_path / task[2].replace('.jpg', '.h5')
            h5f = h5py.File(str(gt_save_path), 'w')
            h5f.create_dataset('density', data=gt_density_map)
            h5f.close()

            amb_save_path = tar_amb_gt_path / task[2].replace('.jpg', '.npy')
            np.save(str(amb_save_path), amb_density_map)

            self.lock.acquire()
            self.done_number += 1
            print(f'Thread {processor_id:<2}: [{self.done_number}/{len(self.task_list)}] '
                  f'{img_path} processed. ')
            self.lock.release()

    def start(self):
        for thread in self.threads:
            thread.daemon = True
            thread.start()

        for thread in self.threads:
            thread.join()

    # partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
    @staticmethod
    def gaussian_filter_density(img, points, multiplying_power, mode='gt'):
        """
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

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
                            default=r'/Users/pantianhang/python_data/datasets/FDST')
        parser.add_argument('-t', '--target', help='target data path',
                            default=r'/Users/pantianhang/python_data/datasets/FDST_for_MFA')
        parser.add_argument('-mp', '--multiplying_power', type=float, default=1.5, help='multiplying power of amb mode')
        parser.add_argument('-rs', '--resize', type=str, default='False', help='whether to resize the map')
        parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=0)
        args = parser.parse_args()
        return args


if __name__ == '__main__':
    mp = MultiProcessor()
    mp.start()
