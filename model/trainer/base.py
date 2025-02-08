import abc
import torch
import os.path as osp
from model.logger import Logger

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass  
        
    @abc.abstractmethod
    def final_record(self):
        pass    

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            vl, va, vap = self.evaluate(self.val_loader)
            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(va),  self.train_epoch)
            self.logger.add_scalar('val_vap', float(vap),  self.train_epoch)
            print('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, vl, va, vap))

            

            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')

            print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))

    def try_logging(self, total_loss, train_acc):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('\n epoch {}, train {:06g}/{:06g}, total loss={:.4f}, acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          total_loss.item(), train_acc.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', total_loss.item(), self.train_step)
            self.logger.add_scalar('train_acc',  train_acc.item(), self.train_step)
            self.logger.dump()

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
