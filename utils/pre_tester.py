import torch
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
import functools

from utils.utils import Utils
from model.model import CSRNet, ConvNeXtIndoorR
from utils.time_estimator import TimeEstimator


class PreTester(Utils):
    def __init__(self, args):
        super(PreTester, self).__init__()
        print('Initializing tester ... ')
        self.args = args
        self.device = torch.device('cuda')
        self.video_dataset_mode = None if self.args.video_dataset_mode == 'None' else self.args.video_dataset_mode
        self.time_estimator = TimeEstimator()

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.cuda.manual_seed(self.args.seed)
        torch.multiprocessing.set_sharing_strategy('file_system')

        if self.video_dataset_mode:
            self.test_list = self.get_video_list(Path(self.args.dataset_path),
                                                 self.video_dataset_mode, False)
        else:
            test_path = Path(self.args.dataset_path) / 'test' / 'images'
            self.test_list = [str(path) for path in test_path.glob('*.jpg')]
            self.test_list.sort(key=functools.cmp_to_key(self.cmp))
            self.test_list = [self.test_list]

        self.dif_calculation = None
        self.gen_npy = None if self.args.gen_npy == 'None' else self.args.gen_npy
        self.gen_map = None if self.args.gen_map == 'None' else self.args.gen_map

        self.mask = self.get_roi_mask(self.args.mask_path)

        self.criterion = nn.MSELoss(size_average=False).to(self.device)

        if os.path.isfile(self.args.pth_path):
            print(f'loading network args from {self.args.pth_path} ...')
            checkpoint = torch.load(self.args.pth_path)
            self.net = checkpoint['net'] if 'net' in checkpoint else 'CSRNet'
            self.mode = checkpoint['mode'] if 'mode' in checkpoint else 'amb'
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
            self.model.load_state_dict(checkpoint['model'])
            self.pretrained = checkpoint['pretrained'] if 'pretrained' in checkpoint else False
            if self.mode == 'amb':
                self.multiplying_power = checkpoint['multiplying_power'] if 'multiplying_power' in checkpoint else 1.5
                self.weight_path = checkpoint['weight_path'] if 'weight_path' in checkpoint else 'None'
                self.number = checkpoint['number'] if 'number' in checkpoint else -1
                self.dif_calculation = checkpoint['dif_calculation'] if 'dif_calculation' in checkpoint else 'div'
            print(f'done.')
        else:
            raise Exception(f'args not found at {self.args.pth_path}')

        # if self.mode == 'amb':
        #     if os.system(f'python make_gt.py {self.args.dataset_path} -m amb -mp {self.multiplying_power} '
        #                  f'-wp {self.weight_path} -n {self.number}') != 0:
        #         raise Exception('failed to make amb files.')

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])

        self.model = nn.DataParallel(self.model)
        print('PreTester initializing done.')

    def execute(self):
        print('test begins ... ')
        self.show_hyper_param(mode='pretrain')
        self.pre_test(self.test_list, self.gen_npy, self.gen_map)
