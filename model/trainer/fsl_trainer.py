import os.path as osp
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer
)
from model.utils import (
    pprint, ensure_path,
    Averager, count_acc,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        
        # save running statistics
        running_dict = {}
        for e in self.model.encoder.state_dict():
            if 'running' in e:
                key_name = '.'.join(e.split('.')[:-1])
                if key_name in running_dict:
                    continue
                else:
                    running_dict[key_name] = {}
                # find the position of BN modules
                component = self.model.encoder
                for att in key_name.split('.'):
                    if att.isdigit():
                        component = component[int(att)]
                    else:
                        component = getattr(component, att)
                
                running_dict[key_name]['mean'] = component.running_mean
                running_dict[key_name]['var'] = component.running_var
        self.running_dict = running_dict          
                
                                                
    def prepare_label(self, num):
        # prepare one-hot label
        args = self.args
        label = torch.arange(args.way, dtype=torch.int16).repeat(num)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        return label
    
    def train(self):
        args = self.args
        self.model.train()

        # start FSL training
        label = self.prepare_label(args.query)
        for epoch in range(1, args.max_epoch + 1):
            # initialize the repo with embeddings
            self.train_epoch += 1
            self.model.train()

            train_loss, train_acc = Averager(), Averager()
            self.model.zero_grad()

            for idx, batch in tqdm(enumerate(self.train_loader, 1), total=100):
                self.train_step += 1
                data, gt_label = batch
                    
                gt_label = gt_label[:args.way] # get the ground-truth label of the current episode
                num_unlabeled_and_support = args.unlabeled + args.shot
                unlabeled_and_support = data[:args.way * num_unlabeled_and_support]
                query = data[args.way * num_unlabeled_and_support:]

                logits = self.model(unlabeled_and_support.cuda(), query, self.train_step)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                train_loss.add(loss.item())
                train_acc.add(acc)
                
                loss.backward()
                self.optimizer.step()                  
                self.model.zero_grad()
                    
                self.try_logging(train_loss, train_acc)

            self.lr_scheduler.step()
            print('LOG: Epoch {}: Train Acc: {}: '.format(epoch, acc))
            self.try_evaluate(epoch)

        torch.save(self.trlog, osp.join(args.save_path, 'trlog.txt'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        # record the runing mean and variance before validation
        for e in self.running_dict:
            self.running_dict[e]['mean_copy'] = deepcopy(self.running_dict[e]['mean'])
            self.running_dict[e]['var_copy'] = deepcopy(self.running_dict[e]['var'])
            
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = self.prepare_label(args.query)
        
        for i, batch in enumerate(data_loader, 1):
            data = batch[0]
            num_unlabeled_and_support = args.unlabeled + args.eval_shot
            unlabeled_and_support = data[:args.way * num_unlabeled_and_support]
            query = data[args.eval_way * num_unlabeled_and_support:]
            logits = self.model.forward_eval(unlabeled_and_support.cuda(), query, i)
            for e in self.running_dict:
                self.running_dict[e]['mean'] = deepcopy(self.running_dict[e]['mean_copy'])
                self.running_dict[e]['var'] = deepcopy(self.running_dict[e]['mean_copy'])
                
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            record[i-1, 0] = loss.item()
            record[i-1, 1] = acc
            del data, unlabeled_and_support, query, logits, loss
            torch.cuda.empty_cache()
            
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.model.train()
        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args      
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        # record the runing mean and variance before validation
        for e in self.running_dict:
            self.running_dict[e]['mean_copy'] = deepcopy(self.running_dict[e]['mean'])
            self.running_dict[e]['var_copy'] = deepcopy(self.running_dict[e]['var'])        
        record = np.zeros((600, 2)) # loss and acc
        label = self.prepare_label(args.query)

        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        
        for i, batch in tqdm(enumerate(self.test_loader, 1),total=600):
            data = batch[0]
            num_unlabeled_and_support = args.unlabeled + args.eval_shot
            unlabeled_and_support = data[:args.way * num_unlabeled_and_support]
            query = data[args.eval_way * num_unlabeled_and_support:]
            logits = self.model.forward_eval(unlabeled_and_support.cuda(), query, i)
            for e in self.running_dict:
                self.running_dict[e]['mean'] = deepcopy(self.running_dict[e]['mean_copy'])
                self.running_dict[e]['var'] = deepcopy(self.running_dict[e]['mean_copy'])
                    
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            record[i-1, 0] = loss.item()
            record[i-1, 1] = acc
            del data, unlabeled_and_support, query, logits, loss
            torch.cuda.empty_cache()
            
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
    
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl
    
        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                    self.trlog['test_acc_interval']))
        # np.savetxt("vis/bs/bs_q_acc1.csv", record[:, 1])
        
        return vl, va, vap

    
    def final_record(self):
        # save the best performance in a txt file

        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['test_acc'],
                    self.trlog['test_acc_interval']))       
