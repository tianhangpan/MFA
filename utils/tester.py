import torch
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
import functools

from utils.utils import Utils
from model.model import CSRNet, CSRNetFuse, ConvNeXtIndoorR, ConvNeXtFuse
from utils.time_estimator import TimeEstimator


class Tester(Utils):
    def __init__(self, args):
        super(Tester, self).__init__()
        print('Initializing tester ... ')
        self.args = args
        self.device = torch.device('cuda')
        self.d_min = 1.0
        self.d_max = 1.5
        self.multiplying_power = None
        self.dif_calculation = None
        self.gen_npy = None if self.args.gen_npy == 'None' else self.args.gen_npy
        self.gen_map = None if self.args.gen_map == 'None' else self.args.gen_map
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

        self.mask = self.get_roi_mask(self.args.mask_path)

        if os.path.isfile(self.args.pth_path):
            print(f'loading data from {self.args.pth_path} ...')
            checkpoint = torch.load(self.args.pth_path)
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
            try:
                self.counter.load_state_dict(checkpoint['counter'])
            except:
                self.counter = ConvNeXtIndoorR(in_channels=3, mode=4).to(self.device)
                self.counter.load_state_dict(checkpoint['counter'])
            self.amb_model.load_state_dict(checkpoint['amb_model'])
            self.fuse_model.load_state_dict(checkpoint['fuse_model'])
            self.d_min = checkpoint['d_min']
            self.d_max = checkpoint['d_max']
            self.amb_supervise = checkpoint['amb_supervise'] if 'amb_supervise' in checkpoint else False
            self.dif_calculation = checkpoint['dif_calculation'] if 'dif_calculation' in checkpoint else 'div'
            # self.multiplying_power = checkpoint['multiplying_power']
            print('done.')
        else:
            raise Exception(f'args not found at {self.args.pth_path}')

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])

        self.counter = nn.DataParallel(self.counter)
        self.amb_model = nn.DataParallel(self.amb_model)
        self.fuse_model = nn.DataParallel(self.fuse_model)
        print('Tester initializing done.')

    def execute(self):
        print('test begins ... ')
        self.show_hyper_param()
        self.test(self.test_list, self.gen_npy, self.gen_map)
