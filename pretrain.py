import argparse
import time
import multiprocessing

from utils.pre_trainer import PreTrainer

parser = argparse.ArgumentParser(description='indoor residual regression pytorch')
parser.add_argument('dataset_path', metavar='DATASET', help='path to dataset')
parser.add_argument('save_path', metavar='SAVE', help='the path to save checkpoint')
parser.add_argument('mode', metavar='MODE', help='pretrain counter or amb model, choose "counter" or "amb"')
parser.add_argument('gpu', metavar='GPU', type=str, help='gpu id to use')
parser.add_argument('task', metavar='TASK', type=str, help='task id to use')
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='FDST dataset', default='None')
parser.add_argument('-n', '--net', metavar='NET', type=str,
                    help='CSRNet, ConvNeXt, ConvNeXtSimple, ConvNeXtSimple2, ConvNeXtUpSamp, ConvNeXtLoad, '
                         'ConvNeXtUpSimple, ConvNeXtSmall, ConvNeXtSmallLoad',
                    default='CSRNet')
parser.add_argument('-cs', '--crop_size', metavar='CS', type=int, default=None, help='crop size')
parser.add_argument('-m', '--mask_path', metavar='MASK', type=str, help='roi mask path', default='None')
parser.add_argument('-ld', '--load', metavar='LOAD', default='None',
                    type=str, help='path to the checkpoint')
parser.add_argument('-cn', '--cluster_num', metavar='DM', type=int, help='num of clusters of val list', default=3)
parser.add_argument('-lr', '--lr', metavar='LR', type=float, default=1e-5, help='learning rate')
parser.add_argument('-wd', '--weight_decay', metavar='WD', type=float, default=1e-4, help='weight decay')
parser.add_argument('-bs', '--batch_size', metavar='BS', type=int, default=16, help='batch size')
parser.add_argument('-pf', '--print_freq', metavar='PF', type=int, default=50, help='print frequency')
parser.add_argument('-ne', '--num_epoch', metavar='NE', type=int, default=400, help='num of epoch')
# parser.add_argument('-mp', '--multiplying_power', type=float, default=1.5, help='multiplying power of amb mode')
# parser.add_argument('-wp', '--weight_path', type=str, default='None', help='path of weight map')
# parser.add_argument('-nm', '--number', type=int, default=-1, help='the number of weight map')
parser.add_argument('-dc', '--dif_calculation', metavar='DC', type=str,
                    help='method to calculate the frame dif', default='div')
parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    args.start_epoch = 0
    args.seed = time.time()
    args.num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else int(args.num_workers)

    trainer = PreTrainer(args)
    trainer.execute()
