import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(dir_path, scripts_to_save=None):
    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    # logits_cl = torch.softmax(logits_cl, dim=-1)
    logits = torch.softmax(logits, dim=-1)
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):
    save_path1 = '-'.join([args.title, args.manner, args.dataset, args.backbone_class, '{:02d}w{:02d}s{:02}q{:02}u'.format(args.way, args.shot, args.query, args.unlabeled)])
    save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                           'lr{:.2g}'.format(args.lr),
                           'InLR{}InIter{}'.format(args.gd_lr, args.inner_iters),
                           str(args.lr_scheduler), 
                           't{}'.format(args.temperature), 
                           #'gpu{}'.format(args.gpu) if args.multi_gpu else 'gpu0',
                           # str(time.strftime('%Y%m%d_%H%M%S'))
                           ])
    if args.para_init is not None:
        save_path1 += '-Pre'
    if args.lr_mul > 1:
        save_path2 += '-Mul{}'.format(args.lr_mul)          
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)    
    
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default="mini_1-30")
    parser.add_argument('--manner', type=str, default='full', choices=['transductive', 'inductive'])
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--unlabeled', type=int, default=30)
    parser.add_argument('--backbone_class', type=str, default='Res12',
                        choices=['Res12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', 
                        choices=['MiniImageNet', 'TieredImageNet', 'CUB', 'FC100', 'CIFARFS'])

    # optimization parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='10')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--gpu', default='2')
    
    # frequent parameters
    parser.add_argument('--temperature', type=float, default=0.5)    
    parser.add_argument('--para_init', type=str, default='./saves/initialization/MiniImageNet/Res12-pre.pth')
    parser.add_argument('--gd_lr', default=0.05, type=float,
                        help='The inner learning rate for MAML-Based model')        
    parser.add_argument('--inner_iters', default=3, type=int,
                        help='The inner iterations for MAML-Based model')
    parser.add_argument('--pseudo_train_iter', default=10, type=int,
                        help='The pseudo train iterations for MAML-Based model')      
    
    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='./ablation/checkpoints')
    
    return parser 