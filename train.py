import argparse
import time
import multiprocessing

from utils.trainer import Trainer


parser = argparse.ArgumentParser(description='indoor residual regression pytorch')
parser.add_argument('dataset_path', metavar='DATASET', help='path to dataset')
parser.add_argument('save_path', metavar='SAVE', help='the path to save checkpoint')
parser.add_argument('amb_model_path', metavar='AMP', type=str, help='the path of amb model pth file')
parser.add_argument('counter_model_path', metavar='CMP', type=str, help='the path of counter model pth path')
parser.add_argument('gpu', metavar='GPU', type=str, help='gpu id to use')
parser.add_argument('task', metavar='TASK', type=str, help='task id to use')
parser.add_argument('-m', '--mask_path', metavar='MASK', type=str, help='roi mask path', default='None')
parser.add_argument('-cs', '--crop_size', metavar='CS', type=int, default=None, help='crop size')
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='FDST dataset', default='None')
parser.add_argument('-ld', '--load', metavar='LOAD', default='None',
                    type=str, help='path to the checkpoint')
parser.add_argument('-cn', '--cluster_num', metavar='DM', type=int, help='num of clusters of val list', default=3)
parser.add_argument('-clr', '--counter_lr', metavar='LR', type=float, default=1e-5, help='counter learning rate')
parser.add_argument('-alr', '--amb_lr', metavar='LR', type=float, default=0,
                    help='amb learning rate, -1 to keep same with clr, 0 to fix the weights')
parser.add_argument('-flr', '--fuse_lr', metavar='LR', type=float, default=-1.0,
                    help='fuse learning rate, -1 to keep same with clr')
parser.add_argument('-wd', '--weight_decay', metavar='WD', type=float, default=1e-4, help='weight decay')
parser.add_argument('-bs', '--batch_size', metavar='BS', type=int, default=16, help='batch size')
parser.add_argument('-pf', '--print_freq', metavar='PF', type=int, default=50, help='print frequency')
parser.add_argument('-ne', '--num_epoch', metavar='NE', type=int, default=400, help='num of epoch')
parser.add_argument('-dmin', '--d_min', metavar='DMIN', type=float, help='min value of dynamic range', default=1.0)
parser.add_argument('-dmax', '--d_max', metavar='DMAX', type=float, help='max value of dynamic range', default=1.5)
parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=0)


if __name__ == '__main__':
    args = parser.parse_args()
    args.start_epoch = 0
    args.seed = time.time()
    args.num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else int(args.num_workers)

    trainer = Trainer(args)
    trainer.execute()
