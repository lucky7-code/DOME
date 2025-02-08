import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy
import csv
from model.utils import count_acc

def loss_KL(y_s, y_t):
    T = 0.4
    """KL divergence for distillation"""
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = F.softmax(y_t / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='sum') * (T ** 2) / y_s.shape[0]
    return loss

def Min_Max_scaling(a):
    # Min-Max scaling
    if len(a) == 0:
        return a
    min_a = torch.min(a)
    max_a = torch.max(a)
    n2 = (a - min_a) / (max_a - min_a)

    return n2


def norm_probability(data, distribution_parameters):
    mean, std = distribution_parameters[0], distribution_parameters[1]
    p = torch.exp(-(data - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)

    return p


def update_params(loss, params, step_size=0.5, first_order=False):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph = first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad

    return updated_params


def inner_train_step(encoder, unlabeled_and_support_data, query, args, task_num):
    """ Inner training step procedure. """
    updated_params = OrderedDict(encoder.named_parameters())

    label = torch.arange(args.way).repeat(args.unlabeled + args.shot)
    query_labeld = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
        query_labeld = query_labeld.type(torch.cuda.LongTensor)
        query = query.cuda()
    else:
        label = label.type(torch.LongTensor)

    unlabeled_data = unlabeled_and_support_data[:args.way * args.unlabeled]
    unlabeled_label = label[:args.way * args.unlabeled]
    support_data = unlabeled_and_support_data[args.way * args.unlabeled:]
    support_label = label[args.way * args.unlabeled:]


    # training an task with the support set
    for i in range(2):
        bs_S_feature, pred = encoder(support_data, updated_params, is_embedding = True)
        loss = F.cross_entropy(pred, support_label)
        updated_params = update_params(loss, updated_params, step_size=args.gd_lr)

    with torch.no_grad():
        bs_u_feature, U_pred = encoder(unlabeled_data, updated_params, is_embedding = True)
        U_pred = torch.softmax(U_pred, dim=-1)
        unlabeled_pseudoLabel = torch.argmax(U_pred, dim=-1).detach()
        # t = - U_pred * torch.log(U_pred)
        # t = torch.where(torch.isnan(t), torch.full_like(t, 0), t)
        # U_sh_entropy = torch.sum(t, -1)
        # U_sh_entropy = Min_Max_scaling(U_sh_entropy)
        # selected_index = torch.where(U_sh_entropy < 0.85)

        # selected_unlabeld_data = unlabeled_data[selected_index]
        # selected_U_pred = U_pred[selected_index]
        # selected_pseudoLabel = unlabeled_pseudoLabel[selected_index]
        # selected_unlabeled_label = unlabeled_label[selected_index]
        # selected_pseudolabel_list = [len(*torch.where(selected_pseudoLabel==i)) for i in range(args.way)]

        # if task_num % args.log_interval < 5 :
        
        # task_num = str(task_num)
        # np.savetxt("vis/bs/u_feature_"+task_num+".csv", bs_u_feature.detach().cpu().numpy())
        # np.savetxt("vis/bs/s_feature_"+task_num+".csv", bs_S_feature.detach().cpu().numpy())
        # np.savetxt("vis/bs/u_pesudo_"+task_num+".csv", unlabeled_pseudoLabel.detach().cpu().numpy())
        
        acc_0 = np.mean((unlabeled_pseudoLabel.cpu().numpy() == unlabeled_label.cpu().numpy()).tolist())
        # with open('vis/bs/bs_pesudo_acc1.csv', 'a', newline='') as file:
            # writer = csv.writer(file)
            # writer.writerow([acc])
        #     selected_acc = np.mean((selected_pseudoLabel.cpu().numpy() == selected_unlabeled_label.cpu().numpy()).tolist())
        #     print("\nThe accuracy of the pseudolabel: {0} ".format(str(acc)))

        #     print("The number of selected unlabeled data: {0} ".format(len(selected_pseudoLabel)))

        #     print("The acc of the pseudolabel after seclected: {0} ".format(str(selected_acc)))

        #     print("The number of selected pseudolabels in all classes : {0} ".format(selected_pseudolabel_list) )

        #     print("xxxx")
    
    # bag_data = torch.tensor([]).cuda()
    # bag_label = torch.tensor([]).cuda()
    

    # for i in range(args.way):
    #     # bag
    #     num_Bag = min(selected_pseudolabel_list) 
    #     data_everyClass = torch.where(selected_pseudoLabel==i)[0]
    #     data_everyClass = data_everyClass.cpu().numpy()

    #     bag_index = np.random.choice(data_everyClass, num_Bag, replace=False)        
    #     bag_data = torch.concat([bag_data, selected_unlabeld_data[bag_index]], 0)
    #     bag_label = torch.concat([bag_label, selected_pseudoLabel[bag_index]], 0)

    # mix_x = torch.concat([support_data, bag_data], 0)
    # mix_y = torch.concat([support_label, bag_label], 0)
    mix_x = torch.concat([support_data, unlabeled_data], 0)
    mix_y = torch.concat([support_label, unlabeled_pseudoLabel], 0)
    
    for i in range(20):
        aug_pred = encoder(mix_x, updated_params)
        loss = F.cross_entropy(aug_pred , mix_y.long())
        updated_params = update_params(loss, updated_params, step_size=args.gd_lr)
        with torch.no_grad():
            U_pred = encoder(unlabeled_data, updated_params)
            U_pred = torch.softmax(U_pred, dim=-1)
            unlabeled_pseudoLabel = torch.argmax(U_pred, dim=-1).detach()
            logitis_query = encoder(query, updated_params) / 0.5
            pesudo_acc = np.mean((unlabeled_pseudoLabel.cpu().numpy() == unlabeled_label.cpu().numpy()).tolist())
            acc = count_acc(logitis_query, query_labeld)
            with open('vis/data/bs_pesudo_acc.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([pesudo_acc])
            with open('vis/data/bs_test_acc.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([acc])
    print(acc_0)
            
        

    return updated_params

class MAML_full(nn.Module):

    def __init__(self, args):
        super().__init__()
       
        if args.backbone_class == 'Res12':
            from model.networks.res12_maml import ResNet12_FSL_FULL
            self.encoder = ResNet12_FSL_FULL(args, args.way, dropblock_size=args.dropblock_size)
            self.classier_aug = Classifer(args, 5)
        else:
            raise ValueError('')
        self.args = args

    def forward(self, data_shot, data_query, task_num):
        # update with gradient descent
        data_query = data_query.cuda()
        updated_params = inner_train_step(self.encoder, data_shot, data_query, self.args, task_num)

        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis

    def forward_eval(self, data_shot, data_query, task_num):
        # update with gradient descent
        data_query = data_query.cuda()
        self.train()
        updated_params = inner_train_step(self.encoder, data_shot, data_query, self.args, task_num)

        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis_query
class Classifer(nn.Module):
    def __init__(self, args, class_num):
        super(Classifer, self).__init__()
        self.dim = 640
        if args.dataset=='CUB':
            self.dim = 512
        self.class_num = class_num
        self.fc = nn.Linear(self.dim, class_num)

    def forward(self, x, params):
        logits = F.linear(x, weight=params['fc.weight'], bias=params['fc.bias'])
        return logits