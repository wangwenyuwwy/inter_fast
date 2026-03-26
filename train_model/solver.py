import os
import time
import gc
from itertools import chain

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader1 import get_loader
from network import *
from myModel import DSC_Net, EnhancedFeatureExtractor

torch.backends.cudnn.deterministic = True


class Solver(object):
    def __init__(self, config, gpus):
        self.device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
        self.cuSize = config.cuSize
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.model_path = config.model_path
        self.label_root = config.label_root
        self.crop_root = config.crop_root
        self.train_budget = config.train_budget
        self.validation_interval = max(1, config.validation_interval)
        self.max_validation_batches = max(0, config.max_validation_batches)
        self.gpus = gpus
        self.build_model()

    def load_model(self):
        def my_load_state_dict(module, state_dict):
            from collections import OrderedDict
            if 'module.' in list(state_dict.keys())[0]:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                module.load_state_dict(new_state_dict, strict=True)
            else:
                module.load_state_dict(state_dict, strict=False)

        for i, module in enumerate(self.cuSize_list):
            model_path = os.path.join(self.model_path, str(self.cuSize), 'module-%d.pkl' % i)
            my_load_state_dict(module, torch.load(model_path, map_location=self.device))

    def build_model(self):
        self.res = []
        if self.cuSize <= 4:
            self.res.append(single_conv(ch_in=1, ch_out=16))
        else:
            self.res.append(single_conv(ch_in=3, ch_out=16))

        self.res.append(EnhancedFeatureExtractor(dim=16))

        if self.cuSize % 5 == 0:
            self.subnet = DSC_Net(out_dim=6, spatial_size=(32, 32), qp_size=4)
        elif self.cuSize == 1:
            self.subnet = DSC_Net(out_dim=6, spatial_size=(16, 16), qp_size=16)
        elif self.cuSize == 2:
            self.subnet = DSC_Net(out_dim=6, spatial_size=(16, 32), qp_size=8)
        elif self.cuSize == 3:
            self.subnet = DSC_Net(out_dim=6, spatial_size=(8, 32), qp_size=8)
        elif self.cuSize == 4:
            self.subnet = DSC_Net(out_dim=6, spatial_size=(8, 16), qp_size=12)
        else:
            self.subnet = DSC_Net(out_dim=6, spatial_size=(32, 32), qp_size=4)

        self.cuSize_list = [self.res[0], self.res[1], self.subnet]
        self.optimizer = optim.Adam(
            chain(self.res[0].parameters(), self.res[1].parameters(), self.subnet.parameters()),
            self.lr,
            [self.beta1, self.beta2],
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.6, patience=4)

        for i in range(2):
            self.res[i] = self.res[i].to(self.device)
        self.subnet = self.subnet.to(self.device)

    def save_model(self):
        model_dir = os.path.join(self.model_path, str(self.cuSize))
        os.makedirs(model_dir, exist_ok=True)
        for i, module in enumerate(self.cuSize_list):
            tmp_model_path = os.path.join(model_dir, 'module-%d.pkl' % i)
            torch.save(module.state_dict(), tmp_model_path)

    def calculate_loss(self, images, pre, total_acc, total_k2_acc, total_remains, gt_remains, thres=0.1):
        images['gt'] = images['gt'].to(self.device)
        sum_loss = 0

        soft = torch.nn.functional.softmax(pre[:, 0:6], dim=1)
        tmp_sum = torch.sum(torch.log(soft) * images['gt'])
        if torch.isnan(tmp_sum):
            sum_loss -= torch.sum(torch.log(soft + 1e-14) * images['gt'])
        else:
            sum_loss -= tmp_sum

        for j in range(images['image'].size(0)):
            if torch.argmax(soft[j]) == torch.argmax(images['gt'][j]):
                total_acc += 1

            for k in range(6):
                if soft[j][k] > thres:
                    total_k2_acc += images['gt'][j][k]
                    total_remains += 1

        sum_loss /= images['image'].size(0)
        return total_acc, total_k2_acc, total_remains, sum_loss, gt_remains

    def run(self, images, total_acc, total_k2_acc, total_length, total_remains, gt_remains, thres=0.1):
        pre = self.res[0](images['image'].to(self.device))
        pre = self.res[1](pre)
        pre = self.subnet(pre, images['qp'].to(self.device))
        total_acc, total_k2_acc, total_remains, sum_loss, gt_remains = self.calculate_loss(
            images, pre, total_acc, total_k2_acc, total_remains, gt_remains, thres
        )
        total_length += images['image'].size(0)
        return total_acc, total_k2_acc, total_length, total_remains, sum_loss, gt_remains

    def validate(self, valid_loader, thres=0.1):
        with torch.no_grad():
            for module in self.cuSize_list:
                module.train(False)
                module.eval()

            total_acc = 0.0
            total_length = 0
            total_remains = 0
            epoch_sum_loss = 0.0
            total_k2_acc = 0.0
            gt_remains = 0.0
            batch_count = 0

            for i, images in enumerate(valid_loader):
                total_acc, total_k2_acc, total_length, total_remains, sum_loss, gt_remains = self.run(
                    images, total_acc, total_k2_acc, total_length, total_remains, gt_remains, thres
                )
                epoch_sum_loss += sum_loss.item()
                batch_count += 1
                if self.max_validation_batches and batch_count >= self.max_validation_batches:
                    break

            denom = max(1, total_length)
            print('[Validation]cuSize:%d, Sum_Loss: %.4f, acc: %.4f, k2_acc: %.4f, remains: %.4f, length: %d, gt_remains: %.4f\n' % (
                self.cuSize,
                epoch_sum_loss / denom * self.batch_size,
                total_acc / denom,
                total_k2_acc / denom,
                total_remains / denom,
                total_length,
                gt_remains / denom,
            ))
            for module in self.cuSize_list:
                module.train(True)

        self.scheduler.step(epoch_sum_loss / max(1, batch_count))
        print(self.optimizer.param_groups[0]['lr'])
        return total_acc, total_k2_acc, total_length, total_remains

    def train(self):
        train_loader = get_loader(
            cuSize=self.cuSize,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            mode='train',
            label_root=self.label_root,
            crop_root=self.crop_root,
        )
        valid_loader = get_loader(
            cuSize=self.cuSize,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            mode='valid',
            label_root=self.label_root,
            crop_root=self.crop_root,
        )

        total_remains = 0
        total_acc = 0.0
        total_length = 0
        epoch_sum_loss = 0.0
        total_k2_acc = 0.0
        gt_remains = 0.0
        start = time.time()
        step_idx = 0
        remaining_steps = self.train_budget
        train_iter = iter(train_loader)

        while remaining_steps > 0:
            for module in self.cuSize_list:
                module.train(True)

            try:
                images = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                print('epoch_end')
                continue

            total_acc, total_k2_acc, total_length, total_remains, sum_loss, gt_remains = self.run(
                images, total_acc, total_k2_acc, total_length, total_remains, gt_remains
            )
            epoch_sum_loss += sum_loss.item()

            self.optimizer.zero_grad()
            sum_loss.backward()
            self.optimizer.step()

            remaining_steps -= 1
            step_idx += 1

            should_validate = (step_idx % self.validation_interval == 0) or remaining_steps == 0
            if should_validate:
                print('Training step {} remaining {} time {}: '.format(step_idx, remaining_steps, time.time() - start))
                denom = max(1, total_length)
                print('[Training]cuSize:%d, Step [%d], Sum_Loss: %.4f, acc: %.4f, k2_acc: %.4f, remains: %.4f,length: %d, gt_remains: %.4f\n' % (
                    self.cuSize,
                    step_idx,
                    epoch_sum_loss,
                    total_acc / denom,
                    total_k2_acc / denom,
                    total_remains / denom,
                    total_length,
                    gt_remains / denom,
                ))
                self.validate(valid_loader)
                self.save_model()
                total_remains = 0
                total_acc = 0.0
                total_length = 0
                epoch_sum_loss = 0.0
                total_k2_acc = 0.0
                gt_remains = 0.0
                start = time.time()

        gc.collect()