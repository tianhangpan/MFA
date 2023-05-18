import argparse
import time
import multiprocessing

from utils.tester import Tester


parser = argparse.ArgumentParser(description='Indoor Residual regression pytorch')
parser.add_argument('dataset_path', metavar='DATASET', help='path to dataset')
parser.add_argument('pth_path', metavar='PTH', help='path to pth file')
parser.add_argument('gpu', metavar='GPU', type=str, help='gpu id to use')
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='FDST dataset', default='None')
parser.add_argument('-m', '--mask_path', metavar='MASK', type=str, help='roi mask path', default='None')
parser.add_argument('-gn', '--gen_npy', metavar='GN', type=str, help='the path to save counts files', default='None')
parser.add_argument('-gm', '--gen_map', metavar='GM', type=str, help='the path to save density maps', default='None')
parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=0)


if __name__ == '__main__':
    args = parser.parse_args()
    args.seed = time.time()
    args.num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else int(args.num_workers)
    tester = Tester(args)
    tester.execute()
