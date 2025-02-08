import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler
from collections import Counter
from model.models.maml_classifer_update import MAML_semi
from model.models.maml_nosiy_label import MAML_full

def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CIFARFS':
        from model.dataloader.cifarfs import CIFARFS as Dataset
        args.dropblock_size = 2                
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import FC100 as Dataset      
        args.dropblock_size = 2        
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet_raw import tieredImageNet as Dataset    
        args.dropblock_size = 5        
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch
    num_workers = args.num_workers * num_device
    
    trainset = Dataset('train', args)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                      num_episodes, args.way,
                                      args.unlabeled + args.shot + args.query, args=args)
    train_loader = DataLoader(dataset=trainset,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)
    
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                                    args.num_eval_episodes, args.eval_way,
                                    args.unlabeled + args.eval_shot + args.eval_query, args=args)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                                     600, args.eval_way,
                                     args.unlabeled + args.eval_shot + args.eval_query, args=args)
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)        

    return train_loader, val_loader, test_loader


def prepare_model(args):
        
    if args.manner == "semi":
        model = MAML_semi(args)
    else:
        model = MAML_full(args)

    # load pre-trained model (no FC weights)
    if args.para_init is not None:
        model_dict = model.state_dict()
        print(model_dict.keys()) 
        # pretrained_dict = torch.load(args.para_init, map_location='cpu')['state_dict']
        # pretrained_dict = {k.replace('base', 'encoder'): v for k, v in pretrained_dict.items() if k.startswith('base')}

        pretrained_dict = torch.load(args.para_init, map_location='cpu')['params'] # map_location=torch.device('cpu')
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(args.gpu.split(',')) > 1:
        gpus = list(map(int, args.gpu.split(',')))
        model = nn.DataParallel(model,device_ids=gpus,dim=0).to(device)
        model = model.module
    else:
        model = model.to(device)

    return model

def prepare_optimizer(model, args):
    if args.lr_mul > 1:
        top_para = [v for k,v in model.named_parameters() if 'encoder' not in k]        
        optimizer = optim.SGD(
                        [{'params': model.encoder.parameters()},
                         {'params': top_para, 'lr': args.lr * args.lr_mul}],
                        lr=args.lr,
                        momentum=args.mom,
                        nesterov=True,
                        weight_decay=args.weight_decay
                    )
       
    else:
        optimizer = optim.SGD(
                        model.parameters(),
                        lr=args.lr,
                        momentum=args.mom,
                        nesterov=True,
                        weight_decay=args.weight_decay
                    )
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
