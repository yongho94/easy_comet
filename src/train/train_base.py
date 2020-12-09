from tqdm import tqdm
import torch
from torch import flatten
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss


def get_trainer(config, device, loader, tb_writer, mode='train'):
    return Trainer(config, device, loader, tb_writer, mode=mode)


class Trainer:
    def __init__(self, config, device, loader, tb_writer, global_step=0, mode='train'):
        self.config = config
        self.device = device
        self.loader = loader
        self.writer = tb_writer
        self.global_step = global_step
        self.mode = mode
        self.opt = None
        self.lr_schedule = None
        self.mean_loss = None
        self.loss_fn = CrossEntropyLoss(reduction='none')

    def init_opt(self, opt):
        self.opt = opt

    def init_lr_schedule(self, lr_schedule):
        self.lr_schedule = lr_schedule

    def train_epoch(self, model, epoch, global_step=None):
        if self.mode == 'train':
            model.train()
        else:
            model.eval()
        model.to(self.device)
        loss_save = list()

        for data in tqdm(self.loader, desc="Epoch : {}".format(epoch)):
            input_ = data['input'].to(self.device)
            seq_mask_ = data['seq_mask'].to(self.device)
            output_ = data['output'].to(self.device)
            loss_mask_ = data['loss_mask'].to(self.device)
            pred_score, pred_prob = model(input_, seq_mask_)
            loss = self._calculate_loss(pred_score, output_, loss_mask_)

            if self.mode == 'train':
                self._optimize(model, loss)
                self.global_step += 1
                self.write_tb(loss, global_step)
            else:
                loss_save.append(loss.item())

        if self.mode != 'train':
            loss = sum(loss_save) / len(loss_save)
            self.write_tb(loss, global_step)
            return loss

    def write_tb(self, loss, global_step):
        if self.mode == 'train':
            lr = self.opt.param_groups[0]['lr']
            self.writer.add_scalar('train/loss', loss, self.global_step)
            self.writer.add_scalar('train/lr', lr, self.global_step)
        else:
            assert global_step is not None
            self.writer.add_scalar('valid/loss', loss, global_step)
        self.writer.flush()

    def _optimize(self, model, loss):
        self.opt.zero_grad()
        loss.backward()
        self.lr_schedule(self.global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
        self.opt.step()

    def _calculate_loss(self, pred_score, label, loss_mask_=None):
        loss = self.loss_fn(flatten(pred_score, 0, 1), flatten(label))
        loss = loss.view(label.size()[0], -1)
        loss = loss * loss_mask_ if loss_mask_ is not None else loss
        loss = torch.sum(loss, dim=1) / torch.sum(loss_mask_, dim=1)
        return torch.mean(loss)


def get_optimizer(model, config):
    return torch.optim.Adam(model.parameters(),
                            lr=1e-05,
                            weight_decay=config.train.weight_decay)


def get_lr_scheduler(optimizer, config, total_steps):
    return WarmupLinearScheduler(optimizer, config, total_steps)


class WarmupLinearScheduler(object):
    def __init__(self, optimizer, config, total_steps):
        self.optimizer = optimizer
        self.lr = config.train.init_lr
        self.total_steps = total_steps
        self.warmup_steps = config.train.warmup_steps
        self.max_lr = config.train.max_lr
        self.warmup_increase = None
        self.lr_decay = None
        self._calculate()
        self._adapt_lr()

    def __call__(self, step):
        if step < self.warmup_steps:
            self.lr += self.warmup_increase
        else:
            self.lr -= self.lr_decay
        assert not self.lr < 0
        self._adapt_lr()

    def _calculate(self):
        self.warmup_increase = self.max_lr / self.warmup_steps
        self.lr_decay = self.max_lr / (self.total_steps - self.warmup_steps)

    def _adapt_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr
