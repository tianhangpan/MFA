import torch
import shutil
import re
import numpy as np
from pathlib import Path
from itertools import chain
import functools

from torch.utils.data import DataLoader
from dataset.dataset import PresentDataset, FrontBackDataset
import pytorch_ssim


class Utils:
    def __init__(self):
        super().__init__()
        self.args = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.optimizer_counter = None
        self.optimizer_amb = None
        self.optimizer_fuse = None
        self.transform = None
        self.mask = None
        self.criterion = None
        self.amb_model = None
        self.counter = None
        self.fuse_model = None
        self.d_min = None
        self.d_max = None
        # self.multiplying_power = None
        self.net = None
        self.mode = None
        self.amb_lr = None
        self.counter_lr = None
        self.fuse_lr = None
        # self.amb_supervise = None
        self.dif_calculation = None
        self.video_dataset_mode = None
        self.crop_size = None
        self.time_estimator = None

    @staticmethod
    def save_checkpoint(state, is_best, task_id, save_path, filename='checkpoint.pth.tar'):
        torch.save(state, save_path + '/' + task_id + filename)
        if is_best:
            shutil.copyfile(save_path + '/' + task_id + filename, save_path + '/' + task_id + 'model_best.pth.tar')

    @staticmethod
    def cmp(x, y):
        a = int(re.findall(r'(\d+)\.\w+$', str(x))[0])
        b = int(re.findall(r'(\d+)\.\w+$', str(y))[0])
        return -1 if a < b else 1

    @staticmethod
    def divide_train_list(train_list, num):
        val_list = []
        new_train_list = []
        val_ele_len = len(train_list) // (10 * num)
        divide_points = [(len(train_list) * i) // num for i in range(num + 1)]
        for i in range(num):
            val_list.append(train_list[divide_points[i + 1] - val_ele_len: divide_points[i + 1]])
            new_train_list.append(train_list[divide_points[i]: divide_points[i + 1] - val_ele_len])
        return new_train_list, val_list

    @staticmethod
    def get_video_list(dataset_path: Path, mode, train=False):
        train_list = None
        test_list = None
        if mode.lower() == 'ucsd':
            if train:
                train_list = [3, 4, 5, 6]
            else:
                test_list = [0, 1, 2, 7, 8, 9]
        else:
            video_list = [list(range(i * 5 + 1, i * 5 + 6)) for i in range(20)]
            if mode.lower() != 'full':
                mode = int(mode)
                if mode in [17, 18]:
                    video_list = video_list[16: 18]
                else:
                    video_list = [video_list[mode]]
            if train:
                train_list = [ls[:3] for ls in video_list]
                train_list = list(chain(*train_list))
            else:
                test_list = [ls[3:] for ls in video_list]
                test_list = list(chain(*test_list))
        if train:
            train_path = dataset_path / 'train'
            train_path_list = [list((train_path / str(video_number) / 'images').glob('*.jpg'))
                               for video_number in train_list]
            for ls in train_path_list:
                for i in reversed(list(range(len(ls)))):
                    if ls[i].name[0] == '.':
                        del ls[i]
            for ls in train_path_list:
                ls.sort(key=functools.cmp_to_key(Utils.cmp))
            video_len = len(train_path_list[0])
            divide_point = video_len * 9 // 10
            val_path_list = [e[divide_point:] for e in train_path_list]
            train_path_list = [e[:divide_point] for e in train_path_list]
            return train_path_list, val_path_list
        else:
            test_path = dataset_path / 'test'
            test_path_list = [list((test_path / str(video_number) / 'images').glob('*.jpg'))
                              for video_number in test_list]
            for ls in test_path_list:
                for i in reversed(list(range(len(ls)))):
                    if ls[i].name[0] == '.':
                        del ls[i]
            for ls in test_path_list:
                ls.sort(key=functools.cmp_to_key(Utils.cmp))
            return test_path_list

    @staticmethod
    def get_trans_list(dataset_path: Path, mode):
        if mode.lower() == 'ucsd':
            train_list = [3, 4, 5, 6]
            train_path = dataset_path / 'train'
            train_path_list = [list((train_path / str(video_number) / 'images').glob('*.jpg'))
                               for video_number in train_list]
            for ls in train_path_list:
                for i in reversed(list(range(len(ls)))):
                    if ls[i].name[0] == '.':
                        del ls[i]
            for ls in train_path_list:
                ls.sort(key=functools.cmp_to_key(Utils.cmp))
            train_path_list = [train_path_list[0][:12], train_path_list[1][:12],
                               train_path_list[2][:12], train_path_list[3][:14]]
        elif mode.lower() == 'mall':
            train_path = dataset_path / 'train' / 'images'
            train_list = [str(path) for path in train_path.glob('*.jpg')]
            train_list.sort(key=functools.cmp_to_key(Utils.cmp))
            train_list, _ = Utils.divide_train_list(train_list, 1)
            train_path_list = [train_list[0][:50]]
        else:
            raise Exception()
        return train_path_list

    @staticmethod
    def get_roi_mask(mask_path):
        if mask_path != 'None':
            mask = np.load(mask_path)
            mask[mask <= 1e-4] = 0
        else:
            mask = None
        return mask

    def amb_linea_trans(self, amb_map):
        amb_1d = amb_map.view(amb_map.size(0), 1, -1)
        amb_max = torch.max(amb_1d, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        amb_min = torch.min(amb_1d, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        amb_map = (amb_map - amb_min) * (self.d_max - self.d_min) / (amb_max - amb_min) + self.d_min
        return amb_map

    def show_hyper_param(self, mode='train'):
        print(f'dataset: {self.args.dataset_path}')
        if self.video_dataset_mode:
            print(f'dataset mode: {self.video_dataset_mode}')
        print(f'crop size: {self.crop_size}')
        print(f'net: {self.net}')
        if mode == 'pretrain':
            print(f'mode: pretrain, {self.mode}')
        elif mode == 'train':
            print(f'dynamic range: {self.d_min} ~ {self.d_max}')
            # print(f'multiplying power: {self.multiplying_power}')
            # print(f'amb supervise: {self.amb_supervise}')
        else:
            raise Exception()

    def unpack_data(self, data, amb_mode=False):
        if amb_mode:
            if self.mode == 'amb':
                _, amb_target, _, mask, _, _, diff_f, diff_b, diff_fb = data
                input_data = torch.cat((diff_f.to(self.device), diff_b.to(self.device)), 1)
                amb_target = amb_target.to(self.device)
                mask = mask.to(self.device)
                return input_data, amb_target, mask
            elif self.mode == 'counter':
                img, _, target, mask = data
                img = img.to(self.device)
                target = target.to(self.device)
                mask = mask.to(self.device)
                return img, target, mask
        else:
            img, amb_target, target, mask, _, _, diff_f, diff_b, diff_fb = data
            amb_input = torch.cat((diff_f.to(self.device), diff_b.to(self.device)), 1)
            img = img.to(self.device)
            amb_target = amb_target.to(self.device)
            target = target.to(self.device)
            mask = mask.to(self.device)
            return img, amb_input, amb_target, target, mask

    def pre_train(self, data_list, epoch):
        data_loader = DataLoader(
            FrontBackDataset(
                data_list,
                self.transform,
                self.mask,
                self.dif_calculation,
                crop_size=self.crop_size,
                train=True,
                csr_mode=(self.net in ['CSRNet'])
            ),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        ) if self.mode == 'amb' else DataLoader(
            PresentDataset(
                data_list,
                self.transform,
                self.mask,
                crop_size=self.crop_size,
                train=True,
                csr_mode=(self.net in ['CSRNet'])
            ),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

        self.model.train()

        loss_sum = 0
        mae = 0

        self.time_estimator.simple_mark()

        for i, data in enumerate(data_loader):
            input_data, target, mask = self.unpack_data(data, True)

            res, _ = self.model(input_data)
            res *= mask
            target *= mask
            loss = self.criterion(res, target)
            loss_sum += loss
            mae += abs(res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % self.args.print_freq == 0:
                loss_sum /= self.args.print_freq * self.args.batch_size
                mae /= self.args.print_freq * self.args.batch_size
                print(f'epoch {epoch:<3}: [{i+1:>5}/{len(data_loader)}]batch loss: {loss_sum:<16.13f}'
                      f' mae: {mae:<8.5f} time: {self.time_estimator.query_time_span()}s')
                loss_sum = 0
                mae = 0
                self.time_estimator.simple_mark()

    def pre_test(self, data_list, gen_npy, gen_map):
        data_loader = DataLoader(
            FrontBackDataset(
                data_list,
                self.transform,
                self.mask,
                self.dif_calculation,
                crop_size=None,
                train=False,
                csr_mode=(self.net in ['CSRNet'])
            ),
            batch_size=1,
            num_workers=self.args.num_workers,
        ) if self.mode == 'amb' else DataLoader(
            PresentDataset(
                data_list,
                self.transform,
                self.mask,
                crop_size=None,
                train=False,
                csr_mode=(self.net in ['CSRNet'])
            ),
            batch_size=1,
            num_workers=self.args.num_workers,
        )

        self.model.eval()

        mse = 0
        mae = 0
        r_mse = 0
        ssim = 0

        model_counts = []
        gt_counts = []

        model_root_path = None
        if gen_map:
            path = Path(gen_map)
            model_root_path = path / 'convnext'
            model_root_path.mkdir(exist_ok=True)

        self.time_estimator.simple_mark()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                print(f' {((i + 1) / len(data_loader)) * 100:.1f}% ...\r', end='')
                input_data, target, mask = self.unpack_data(data, True)

                res, _ = self.model(input_data)
                res *= mask
                target *= mask

                mse += self.criterion(res, target).item()
                mae += abs(res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device))
                r_mse += (res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device)) ** 2
                ssim += pytorch_ssim.ssim(res, target).item()

                if gen_npy:
                    model_counts.append(round(res.data.sum().item()))
                    gt_counts.append(round(target.sum().item()))

                if gen_map:
                    model_map = np.array(res.cpu())
                    model_map = np.reshape(model_map, model_map.shape[2:])
                    model_path = model_root_path / f'{i}.npy'
                    np.save(str(model_path), model_map)

        mae /= len(data_loader)
        mse /= len(data_loader)
        r_mse /= len(data_loader)
        r_mse = float(r_mse) ** .5
        ssim /= len(data_loader)

        if gen_npy:
            model_counts = np.array(model_counts)
            gt_counts = np.array(gt_counts)
            model_path = gen_npy + f'/convnext_counts.npy'
            gt_path = gen_npy + f'/gt_counts.npy'
            np.save(model_path, model_counts)
            np.save(gt_path, gt_counts)

        print(f' MAE: {mae:.5f}')
        print(f' MSE: {mse:.5f}')
        print(f' RMSE: {r_mse:.5f}')
        print(f' SSIM: {ssim: .5f}')
        print(f' Time: {self.time_estimator.query_time_span()}s')

        return mae

    def train(self, data_list, epoch):
        data_loader = DataLoader(
            FrontBackDataset(
                data_list,
                self.transform,
                self.mask,
                self.dif_calculation,
                crop_size=self.crop_size,
                train=True,
                csr_mode=(self.net in ['CSRNet'])
            ),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers
        )

        self.counter.train()
        self.amb_model.train()
        self.fuse_model.train()

        loss_sum = 0
        counter_mae = 0
        final_mae = 0
        # amb_mae = 0

        self.time_estimator.simple_mark()

        for i, data in enumerate(data_loader):
            img, amb_input, amb_target, target, mask = self.unpack_data(data, False)

            _, counter_features = self.counter(img)
            amb_res, amb_features = self.amb_model(amb_input)
            counter_res = self.fuse_model(counter_features, amb_features)
            counter_res *= mask
            target *= mask

            amb_res = self.amb_linea_trans(amb_res)
            # amb_target = self.amb_linea_trans(amb_target)

            # amb_1d = amb_res.view(amb_res.size(0), 1, -1)
            # amb_max = torch.max(amb_1d, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            # amb_min = torch.min(amb_1d, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            # amb_res = (amb_res - amb_min) * (self.d_max - self.d_min) / (amb_max - amb_min) + self.d_min
            final_res = counter_res * amb_res

            loss = self.criterion(final_res, target)
            # if self.amb_supervise:
            #     loss += self.criterion(amb_res, amb_target)
            #     amb_mae += abs(amb_res.data.sum() - amb_target.sum().type(torch.FloatTensor).to(self.device))
            loss_sum += loss
            counter_mae += abs(counter_res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device))
            final_mae += abs(final_res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device))

            self.optimizer_counter.zero_grad()
            if self.amb_lr != 0.0:
                self.optimizer_amb.zero_grad()
            self.optimizer_fuse.zero_grad()
            loss.backward()
            self.optimizer_counter.step()
            if self.amb_lr != 0.0:
                self.optimizer_amb.step()
            self.optimizer_fuse.step()

            if (i + 1) % self.args.print_freq == 0:
                loss_sum /= self.args.print_freq * self.args.batch_size
                counter_mae /= self.args.print_freq * self.args.batch_size
                final_mae /= self.args.print_freq * self.args.batch_size
                # if self.amb_supervise:
                #     amb_mae /= self.args.print_freq * self.args.batch_size
                print(f'epoch {epoch:<3}: [{i + 1:>5}/{len(data_loader)}]batch loss: {loss_sum:<16.13f} ' +
                      f'counter mae: {counter_mae:<8.5f} final mae: {final_mae:<8.5f} '
                      f'time: {self.time_estimator.query_time_span()}s')
                loss_sum = 0
                counter_mae = 0
                final_mae = 0
                # amb_mae = 0
                self.time_estimator.simple_mark()

    def test(self, data_list, gen_npy, gen_map):
        data_loader = DataLoader(
            FrontBackDataset(
                data_list,
                self.transform,
                self.mask,
                self.dif_calculation,
                crop_size=None,
                train=False,
                csr_mode=(self.net in ['CSRNet'])
            ),
            batch_size=1,
            num_workers=self.args.num_workers
        )

        self.counter.eval()
        self.amb_model.eval()
        self.fuse_model.eval()

        counter_rmse = 0
        counter_mae = 0
        final_rmse = 0
        final_mae = 0
        ssim = 0

        model_counts = []
        gt_counts = []

        model_root_path = None
        gt_root_path = None
        if gen_map:
            path = Path(gen_map)
            model_root_path = path / 'model'
            gt_root_path = path / 'gt'
            model_root_path.mkdir(exist_ok=True)
            gt_root_path.mkdir(exist_ok=True)

        self.time_estimator.simple_mark()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                print(f' {((i + 1) / len(data_loader)) * 100:.1f}% ...\r', end='')
                img, amb_input, _, target, mask = self.unpack_data(data, False)

                _, counter_features = self.counter(img)
                amb_res, amb_features = self.amb_model(amb_input)
                counter_res = self.fuse_model(counter_features, amb_features)
                counter_res *= mask
                target *= mask

                amb_res = self.amb_linea_trans(amb_res)
                # amb_1d = amb_res.view(amb_res.size(0), 1, -1)
                # amb_max = torch.max(amb_1d, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
                # amb_min = torch.min(amb_1d, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
                # amb_res = (amb_res - amb_min) * (self.d_max - self.d_min) / (amb_max - amb_min) + self.d_min
                final_res = counter_res * amb_res

                counter_mae += abs(counter_res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device))
                counter_rmse += (counter_res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device)) ** 2
                final_mae += abs(final_res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device))
                final_rmse += (final_res.data.sum() - target.sum().type(torch.FloatTensor).to(self.device)) ** 2
                ssim += pytorch_ssim.ssim(final_res, target).item()

                if gen_npy:
                    model_counts.append(round(final_res.data.sum().item()))
                    gt_counts.append(round(target.sum().item()))

                if gen_map:
                    model_map = np.array(final_res.cpu())
                    model_map = np.reshape(model_map, model_map.shape[2:])
                    gt_map = np.array(target.cpu())
                    gt_map = np.reshape(gt_map, gt_map.shape[2:])
                    model_path = model_root_path / f'{i}.npy'
                    gt_path = gt_root_path / f'{i}.npy'
                    np.save(str(model_path), model_map)
                    np.save(str(gt_path), gt_map)

        counter_mae /= len(data_loader)
        counter_rmse /= len(data_loader)
        counter_rmse = float(counter_rmse) ** .5
        final_mae /= len(data_loader)
        final_rmse /= len(data_loader)
        final_rmse = float(final_rmse) ** .5
        ssim /= len(data_loader)

        if gen_npy:
            model_counts = np.array(model_counts)
            gt_counts = np.array(gt_counts)
            model_path = gen_npy + f'/model_counts.npy'
            gt_path = gen_npy + f'/gt_counts.npy'
            np.save(model_path, model_counts)
            np.save(gt_path, gt_counts)

        print(f' counter MAE: {counter_mae:.5f}')
        print(f' counter MSE: {counter_rmse:.5f}')
        print(f' final MAE: {final_mae:.5f}')
        print(f' final MSE: {final_rmse:.5f}')
        print(f' SSIM: {ssim:.5f}')
        print(f' Time: {self.time_estimator.query_time_span()}s')

        return final_mae
