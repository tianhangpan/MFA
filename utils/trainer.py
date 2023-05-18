import torch
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
import functools

from utils.utils import Utils
from utils.time_estimator import TimeEstimator
from model.model import CSRNet, CSRNetFuse, ConvNeXtIndoorR, ConvNeXtFuse


class Trainer(Utils):
    def __init__(self, args):
        super(Trainer, self).__init__()
        print('Initializing trainer ... ')
        self.device = torch.device('cuda')
        self.args = args
        self.best_pre = 1e9
        self.time_estimator = TimeEstimator()
        self.d_min = self.args.d_min
        self.d_max = self.args.d_max
        # self.multiplying_power = None
        self.load = None if self.args.load == 'None' else self.args.load
        self.counter_lr = self.args.counter_lr
        self.amb_lr = self.counter_lr if self.args.amb_lr == -1.0 else self.args.amb_lr
        self.fuse_lr = self.counter_lr if self.args.fuse_lr == -1.0 else self.args.fuse_lr
        # self.amb_supervise = True if self.args.amb_supervise == 'True' else False
        self.video_dataset_mode = None if self.args.video_dataset_mode == 'None' else self.args.video_dataset_mode
        self.dif_calculation = None
        self.crop_size = self.args.crop_size

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.cuda.manual_seed(self.args.seed)
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.args.save_path = Path(self.args.save_path) / self.args.task
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

        if os.path.isfile(self.args.amb_model_path):
            print(f'loading amb model data from {self.args.amb_model_path}')
            checkpoint = torch.load(self.args.amb_model_path)
            self.net = checkpoint['net']
            if self.net == 'CSRNet':
                self.amb_model = CSRNet(in_channels=6).to(self.device)
                self.counter = CSRNet(in_channels=3).to(self.device)
                self.fuse_model = CSRNetFuse().to(self.device)
            elif self.net == 'ConvNeXt':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=1).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=1).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=1).to(self.device)
            elif self.net == 'ConvNeXtSimple':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=2).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=2).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=2).to(self.device)
            elif self.net == 'ConvNeXtSimple2':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=3).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=3).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=3).to(self.device)
            elif self.net == 'ConvNeXtUpSamp':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=4).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=4).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=4).to(self.device)
            elif self.net == 'ConvNeXtLoad':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=5).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=5).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=5).to(self.device)
            elif self.net == 'ConvNeXtUpSimple':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=6).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=6).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=6).to(self.device)
            elif self.net == 'ConvNeXtBilinearLoad':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=7).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=7).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=7).to(self.device)
            elif self.net == 'ConvNeXtSmall':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=8).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=8).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=8).to(self.device)
            elif self.net == 'ConvNeXtSmallLoad':
                self.amb_model = ConvNeXtIndoorR(in_channels=6, mode=9).to(self.device)
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=9).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=9).to(self.device)
            self.amb_model.load_state_dict(checkpoint['model'])
            # self.multiplying_power = checkpoint['multiplying_power']
            self.dif_calculation = checkpoint['dif_calculation'] if 'dif_calculation' in checkpoint else 'div'
            print('done.')
        else:
            raise Exception(f'amb model data not found at {self.args.amb_model_path}')

        if self.args.counter_model_path == 'Original':
            if self.net == 'ConvNeXtLoad':
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=5).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=5).to(self.device)
        elif self.args.counter_model_path == 'PreTrain':
            if self.net == 'ConvNeXtUpSamp':
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=4).to(self.device)
                self.fuse_model = ConvNeXtFuse(mode=4).to(self.device)
        else:
            if os.path.isfile(self.args.counter_model_path):
                print(f'loading counter model data from {self.args.counter_model_path}')
                checkpoint = torch.load(self.args.counter_model_path)
                # assert self.net == checkpoint['net']
                if checkpoint['net'] == 'ConvNeXtUpSamp':
                    self.counter = ConvNeXtIndoorR(in_channels=3, mode=4).to(self.device)
                key_name = 'state_dict' if 'state_dict' in checkpoint else 'model'
                try:
                    self.counter.load_state_dict(checkpoint[key_name])
                except:
                    map_dict = {'input_end.0.weight': 'frontend.0.weight', 'input_end.0.bias': 'frontend.0.bias',
                                'frontend.0.weight': 'frontend.2.weight', 'frontend.0.bias': 'frontend.2.bias',
                                'frontend.3.weight': 'frontend.5.weight', 'frontend.3.bias': 'frontend.5.bias',
                                'frontend.5.weight': 'frontend.7.weight', 'frontend.5.bias': 'frontend.7.bias',
                                'frontend.8.weight': 'frontend.10.weight', 'frontend.8.bias': 'frontend.10.bias',
                                'frontend.10.weight': 'frontend.12.weight', 'frontend.10.bias': 'frontend.12.bias',
                                'frontend.12.weight': 'frontend.14.weight', 'frontend.12.bias': 'frontend.14.bias',
                                'frontend.15.weight': 'frontend.17.weight', 'frontend.15.bias': 'frontend.17.bias',
                                'frontend.17.weight': 'frontend.19.weight', 'frontend.17.bias': 'frontend.19.bias',
                                'frontend.19.weight': 'frontend.21.weight', 'frontend.19.bias': 'frontend.21.bias'}
                    now_dict = self.counter.state_dict()
                    check_dict = checkpoint[key_name]
                    for key in now_dict:
                        now_dict[key] = check_dict[map_dict[key]] if key in map_dict else check_dict[key]
                    self.counter.load_state_dict(now_dict)
                print('done.')
            else:
                raise Exception(f'counter data not found at {self.args.counter_model_path}')

        self.criterion = nn.MSELoss(size_average=False).to(self.device)
        self.optimizer_counter = torch.optim.AdamW(self.counter.parameters(), self.counter_lr,
                                                   weight_decay=self.args.weight_decay)
        self.optimizer_amb = torch.optim.AdamW(self.amb_model.parameters(), self.amb_lr,
                                               weight_decay=self.args.weight_decay)
        self.optimizer_fuse = torch.optim.AdamW(self.fuse_model.parameters(), self.fuse_lr,
                                                weight_decay=self.args.weight_decay)

        if self.load:
            if os.path.isfile(self.load):
                print(f'loading checkpoint from {self.load} ...')
                checkpoint = torch.load(self.load)
                self.args.start_epoch = checkpoint['epoch']
                self.counter.load_state_dict(checkpoint['counter'])
                self.optimizer_counter.load_state_dict(checkpoint['optimizer_counter'])
                self.amb_model.load_state_dict(checkpoint['amb_model'])
                self.optimizer_amb.load_state_dict(checkpoint['optimizer_amb'])
                self.fuse_model.load_state_dict(checkpoint['fuse_model'])
                self.optimizer_fuse.load_state_dict(checkpoint['optimizer_fuse'])
                # self.best_pre = checkpoint['best_pre']
                self.d_min = checkpoint['d_min'] if 'd_min' in checkpoint else self.d_min
                self.d_max = checkpoint['d_max'] if 'd_max' in checkpoint else self.d_max
                # self.multiplying_power = checkpoint['multiplying_power']
                # self.amb_supervise = checkpoint['amb_supervise'] if 'amb_supervise' in checkpoint else False
                self.dif_calculation = checkpoint['dif_calculation'] if 'dif_calculation' in checkpoint else 'div'
                self.best_pre = checkpoint['best_pre']
            else:
                print(f'checkpoint not found at {self.load}')

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])

        self.amb_model = nn.DataParallel(self.amb_model)
        self.counter = nn.DataParallel(self.counter)
        self.fuse_model = nn.DataParallel(self.fuse_model)
        print('Trainer initializing done.')

    def execute(self):
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            print(f'epoch: {epoch} train begin ... ')
            print(f"counter lr: {self.optimizer_counter.param_groups[-1]['lr']}")
            print(f"amb lr: {self.optimizer_amb.param_groups[-1]['lr']}")
            print(f"fuse lr: {self.optimizer_fuse.param_groups[-1]['lr']}")
            self.show_hyper_param()

            self.time_estimator.mark()
            self.time_estimator.estimate(epoch, self.args.num_epoch)

            self.train(self.train_list, epoch)
            print('validation begins ...')
            precision = self.test(self.val_list, None, None)
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
            postfix = ''
            for tup in map_list:
                if epoch in tup[0]:
                    postfix = tup[1]
                    break
            Utils.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'best_pre': self.best_pre,
                    'net': self.net,
                    'counter': self.counter.module.state_dict(),
                    'amb_model': self.amb_model.module.state_dict(),
                    'fuse_model': self.fuse_model.module.state_dict(),
                    'optimizer_counter': self.optimizer_counter.state_dict(),
                    'optimizer_amb': self.optimizer_amb.state_dict(),
                    'optimizer_fuse': self.optimizer_fuse.state_dict(),
                    'dataset': self.args.dataset_path,
                    'd_min': self.d_min,
                    'd_max': self.d_max,
                    # 'multiplying_power': self.multiplying_power,
                    # 'amb_supervise': self.amb_supervise,
                    'dif_calculation': self.dif_calculation,
                },
                is_best,
                self.args.task + postfix,
                str(self.args.save_path)
            )
