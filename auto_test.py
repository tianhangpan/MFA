import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser(description='Indoor Residual regression pytorch')
parser.add_argument('dataset_path', metavar='DATASET', help='path to dataset')
parser.add_argument('saved_path', metavar='SP', type=str, help='saved path')
parser.add_argument('task_name', metavar='TN', help='task name')
parser.add_argument('gpu', metavar='GPU', type=str, help='gpu id to use')
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='FDST dataset', default='None')
parser.add_argument('-m', '--mask_path', metavar='MASK', type=str, help='roi mask path', default='None')
parser.add_argument('-mo', '--mode', metavar='MODE', type=str, help='pretest or test', default='test')
parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=0)


if __name__ == '__main__':
    args = parser.parse_args()
    saved_path = Path(args.saved_path)
    epoch_nums = ['0~50', '51~100', '101~150', '151~200', '201~250', '250~300', '301~350', '351~400',
                  '401~450', '451~500', '501~550', '551~600', '601~650', '651~700', '701~750', '751~800']
    modes = ['model_best', 'checkpoint']
    for epoch_num in epoch_nums:
        for mode in modes:
            file_name = args.task_name + '_' + epoch_num + mode + '.pth.tar'
            file_path = saved_path / file_name
            if file_path.exists():
                print(f'testing: {file_name}')
                assert os.system(f'python {"test.py" if args.mode == "test" else "pretest.py"} '
                                 f'{args.dataset_path} {str(file_path)} '
                                 f'{args.gpu} -m {args.mask_path} -vd {args.video_dataset_mode} '
                                 f'-nw {args.num_workers}') == 0
