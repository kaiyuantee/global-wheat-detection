import torch
import time
import os
from glob import glob
import numpy as np
from datetime import datetime
from .metrics import calculate_image_precision, AverageMeter


class Fit:

    def __init__(self, model, device, config, weights_path, resume_path=None, resume=False):
        self.config = config
        self.epochs = self.config.epochs
        self.model = model
        self.device = device
        self.base_dir = weights_path
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_loss = 10 ** 5
        self.epoch = 0
        self.resume = resume
        self.resume_path = resume_path

        param_optim = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optim if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optim if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=config.lr)
        self.scheduler = config.Scheduler(self.optimizer, **config.scheduler_params)
        self.log(f'Running on {self.device}')

    def fit(self, train_data_loader, val_data_loader):
        if self.resume:
            self.load_model(self.resume_path)
        for i in range(self.epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLearning Rate: {lr}')

            # train
            t = time.time()
            summary_loss = self.train_one_epoch(train_data_loader)
            self.log(f'TRAIN\n Epoch: {self.epoch}, Loss: {summary_loss.avg:.5f}, '
                     f'Time Spent: {(time.time() - t):.5f}')
            self.save_model(f'{self.base_dir}/lastcheckpoint.bin')

            # evaluate
            t = time.time()
            mean_precision = self.eval(val_data_loader)
            self.log(
                f'VAlIDATION\n Epoch: {self.epoch}, Mean Precision: {mean_precision:.4f}, '
                f'Time Spent: {(time.time() - t):.5f}')

            if summary_loss.avg < self.best_loss:
                self.best_loss = summary_loss.avg
                self.model.eval()
                self.save_model(f'{self.base_dir}/bestcheckpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/bestcheckpoint-*epoch.bin'))[:-3]:
                    os.remove(path)
            # lr scheduler
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    @torch.no_grad()
    def eval(self, val_data_loader):
        mean_precision = []
        iou_thresh = [np.round(x, 2) for x in np.arange(0.5, 0.75, 0.05)]
        detection_threshold = 0.37
        self.model.eval()
        self.model.to(self.device)
        for images, targets, image_ids in val_data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(images)
            for i, image in enumerate(images):
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()

                pred = boxes[scores >= detection_threshold].astype(np.int32)
                scores = scores[scores >= detection_threshold]
                gt = targets[i]['boxes'].data.cpu().numpy()
                sort_idx = np.argsort(scores)[::-1]
                pred_sort = pred[sort_idx]
                precision = calculate_image_precision(gt, pred_sort, thresholds=iou_thresh, form='pascal_voc')
                mean_precision.append(precision)
        return np.mean(mean_precision)

    def train_one_epoch(self, train_data_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for i, (images, target, image_id) in enumerate(train_data_loader):
            if self.config.verbose:
                if i % self.config.verbose_step == 0:
                    print(
                        f'Train Step {i}/{len(train_data_loader)}, Loss: {summary_loss.avg:.5f},'
                        f' Time Spent: {(time.time() - t):.5f}',
                        end='\r')

            images = list(image.to(self.device) for image in images)
            batch_size = len(images)
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in target]
            self.optimizer.zero_grad()
            loss_dict = self.model(images, labels)  # boxes,labels)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            summary_loss.update(losses.detach().item(), batch_size)
            self.optimizer.step()
            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss

    def save_model(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_loss,
            'epoch': self.epoch,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, msg):
        if self.config.verbose:
            print(msg)
        with open(self.log_path, 'a+') as logger:  # write inside a txt file
            logger.write(f'{msg}\n')
