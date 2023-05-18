import torch
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
import functools

from utils.utils import Utils
from utils.time_estimator import TimeEstimator
from model.model import CSRNet, ConvNeXtIndoorR


class PreTrainer(Utils):
    def __init__(self, args):
        super(PreTrainer, self).__init__()
        print('Initializing trainer ... ')
        self.device = torch.device('cuda')
        self.args = args
        self.best_pre = 1e9
        self.time_estimator = TimeEstimator()
        # self.multiplying_power = self.args.multiplying_power
        # self.weight_path = self.args.weight_path
        # self.number = self.args.number
        self.mode = self.args.mode
        self.net = self.args.net
        self.load = None if self.args.load == 'None' else self.args.load
        self.dif_calculation = self.args.dif_calculation
        self.video_dataset_mode = None if self.args.video_dataset_mode == 'None' else self.args.video_dataset_mode
        self.crop_size = self.args.crop_size
        assert self.mode in ['amb', 'counter']
        assert self.net in ['CSRNet', 'ConvNeXt', 'ConvNeXtSimple', 'ConvNeXtSimple2', 'ConvNeXtUpSamp',
                            'ConvNeXtLoad', 'ConvNeXtUpSimple', 'ConvNeXtBilinearLoad', 'ConvNeXtSmall',
                            'ConvNeXtSmallLoad']
        assert self.dif_calculation in ['div', 'sub']

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.cuda.manual_seed(self.args.seed)
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.args.save_path = Path(self.args.save_path) / (self.args.task + f'_pre{self.mode}')
        self.args.save_path.mkdir(exist_ok=True)

        if self.video_dataset_mode:
            self.train_list, self.val_list = Utils.get_video_list(Path(self.args.dataset_path),
                                                                  self.video_dataset_mode, True)
        else:
            train_path = Path(self.args.dataset_path) / 'train' / 'images'
            train_list = [str(path) for path in train_path.glob('*.jpg')]
            train_list.sort(key=functools.cmp_to_key(self.cmp))
            self.train_list, self.val_list = Utils.divide_train_list(train_list, self.args.cluster_num)

        self.mask = self.get_roi_mask(self.args.mask_path)

        in_channels = 3 if self.mode == 'counter' else 6
        if self.net == 'CSRNet':
            self.model = CSRNet(in_channels=in_channels).to(self.device)
        elif self.net == 'ConvNeXt':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=1).to(self.device)
        elif self.net == 'ConvNeXtSimple':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=2).to(self.device)
        elif self.net == 'ConvNeXtSimple2':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=3).to(self.device)
        elif self.net == 'ConvNeXtUpSamp':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=4).to(self.device)
        elif self.net == 'ConvNeXtLoad':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=5).to(self.device)
        elif self.net == 'ConvNeXtUpSimple':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=6).to(self.device)
        elif self.net == 'ConvNeXtBilinearLoad':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=7).to(self.device)
        elif self.net == 'ConvNeXtSmall':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=8).to(self.device)
        elif self.net == 'ConvNeXtSmallLoad':
            self.model = ConvNeXtIndoorR(in_channels=in_channels, mode=9).to(self.device)
        self.criterion = nn.MSELoss(size_average=False).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

        if self.load:
            if os.path.isfile(self.args.load):
                print(f'loading checkpoint from {self.load} ...')
                checkpoint = torch.load(self.load)
                self.args.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pre = checkpoint['best_pre']
                if self.mode == 'amb':
                    self.multiplying_power = checkpoint['multiplying_power'] if 'multiplying_power' in checkpoint \
                        else 1.5
                    # self.weight_path = checkpoint['weight_path'] if 'weight_path' in checkpoint else 'None'
                    # self.number = checkpoint['number'] if 'number' in checkpoint else -1
                self.dif_calculation = checkpoint['dif_calculation'] if 'dif_calculation' in checkpoint else 'div'
                print(f'successfully loaded checkpoint, epoch: {self.args.start_epoch}')
            else:
                print(f'checkpoint not found at {self.load}')

        # if self.mode == 'amb':
        #     if os.system(f'python make_gt.py {self.args.dataset_path} -m amb -mp {self.multiplying_power} '
        #                  f'-wp {self.weight_path} -n {self.number}') != 0:
        #         raise Exception('failed to make amb files.')

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])

        self.model = nn.DataParallel(self.model)
        print('PreTrainer initializing done.')

    def execute(self):
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            print(f'epoch: {epoch} train begin ... ')
            print(f"lr: {self.optimizer.param_groups[-1]['lr']}")
            self.show_hyper_param(mode='pretrain')

            self.time_estimator.mark()
            self.time_estimator.estimate(epoch, self.args.num_epoch)

            self.pre_train(self.train_list, epoch)
            print('validation begins ...')
            precision = self.pre_test(self.val_list, None, None)
            is_best = precision < self.best_pre
            self.best_pre = min(precision, self.best_pre)
            print(f' * best MAE: {self.best_pre:.5f}')

            map_list = [(list(range(0, 51)), '_0~50'), (list(range(51, 101)), '_51~100'),
                        (list(range(101, 151)), '_101~150'), (list(range(151, 201)), '_151~200'),
                        (list(range(201, 251)), '_201~250'), (list(range(251, 301)), '_251~300'),
                        (list(range(301, 351)), '_301~350'), (list(range(351, 401)), '_351~400'),
                        (list(range(401, 451)), '_401~450'), (list(range(451, 501)), '_451~500'),
                        (list(range(501, 551)), '_501~550'), (list(range(551, 601)), '_551~600'),
                        (list(range(601, 651)), '_601~650'), (list(range(651, 701)), '_651~700'),
                        (list(range(701, 751)), '_701~750'), (list(range(751, 801)), '_751~800')]
            postfix = '_pre' + self.mode
            for tup in map_list:
                if epoch in tup[0]:
                    postfix += tup[1]
                    break
            Utils.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'best_pre': self.best_pre,
                    'model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'dataset': self.args.dataset_path,
                    # 'multiplying_power': self.multiplying_power,
                    # 'weight_path': self.args.weight_path,
                    # 'number': self.args.number,
                    # new
                    'video_dataset_mode': self.video_dataset_mode,
                    'mode': self.mode,
                    'net': self.args.net,
                    'dif_calculation': self.dif_calculation,
                },
                is_best,
                self.args.task + postfix,
                str(self.args.save_path)
            )
