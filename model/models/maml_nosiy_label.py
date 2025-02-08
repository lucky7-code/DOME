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


def inner_train_step_mine(encoder, classier_aug, unlabeled_and_support_data, query, args, phase, task_num):
    """ Inner training step procedure. """
    updated_params_en = OrderedDict(encoder.named_parameters())
    updated_params_cl = OrderedDict(classier_aug.named_parameters())

    if args.manner == 'transductive':
        label = torch.arange(args.way).repeat(args.unlabeled + args.shot + args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        unlabeled_and_support_data = torch.concat([query.cuda(), unlabeled_and_support_data], 0)
        unlabeled_data = unlabeled_and_support_data[:args.way * (args.unlabeled + args.query)]
        unlabeled_label = label[:args.way * (args.unlabeled + args.query)]
        support_data = unlabeled_and_support_data[args.way * (args.unlabeled + args.query):]
        support_label = label[args.way * (args.unlabeled + args.query):]
    
    else:
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

    # mini_epoch
    for i in range(20):
        if i == 0:
            mix_x = support_data
            mix_y = support_label
        for _ in range(5):    
            pred = encoder(mix_x, updated_params_en)
            loss = F.cross_entropy(pred, mix_y.long())
            updated_params_en = update_params(loss, updated_params_en, step_size=args.gd_lr)

        # generate the noisy labels
        with torch.no_grad():
            our_u_feature, U_pred = encoder(unlabeled_data, updated_params_en, is_embedding = True)
            U_pred = torch.softmax(U_pred, dim=-1)
            unlabeled_pseudoLabel = torch.argmax(U_pred, dim=-1).detach() # noisy label

            # using the entropy to make a simplly select for noisy label
            t = - U_pred * torch.log(U_pred)
            t = torch.where(torch.isnan(t), torch.full_like(t, 0), t)
            U_sh_entropy = torch.sum(t, -1)
            U_sh_entropy = Min_Max_scaling(U_sh_entropy)
            selected_index = torch.where(U_sh_entropy < (0.9 - 0.015 * i))

            selected_unlabeld_data = unlabeled_data[selected_index]
            selected_U_pred = U_sh_entropy[selected_index]
            selected_pseudoLabel = unlabeled_pseudoLabel[selected_index]
            selected_unlabeled_label = unlabeled_label[selected_index]
            selected_pseudolabel_list = [len(*torch.where(selected_pseudoLabel==i)) for i in range(args.way)]
            pesudo_acc = np.mean((unlabeled_pseudoLabel.cpu().numpy() == unlabeled_label.cpu().numpy()).tolist())
            
            # if i == (args.pseudo_train_iter - 1):
            #     task_num = str(task_num)
            #     our_S_feature, pred = encoder(support_data, updated_params_en, is_embedding = True)
            #     np.savetxt("vis/our_u_feature_"+task_num+".csv", our_u_feature.detach().cpu().numpy())
            #     np.savetxt("vis/our_s_feature_"+task_num+".csv", our_S_feature.detach().cpu().numpy())
            #     np.savetxt("vis/our_u_pesudo_"+task_num+".csv", unlabeled_pseudoLabel.detach().cpu().numpy())
            
                
                
            #     # if phase=='train' :
            #     acc = np.mean((unlabeled_pseudoLabel.cpu().numpy() == unlabeled_label.cpu().numpy()).tolist())
            #     with open('vis/pesudo_acc.csv', 'a', newline='') as file:
            #         writer = csv.writer(file)
            #         writer.writerow([acc])
            
            # selected_acc = np.mean((selected_pseudoLabel.cpu().numpy() == selected_unlabeled_label.cpu().numpy()).tolist())
            # print("\nThe step {1} :\nThe accuracy of the pseudolabel: {0} ".format(str(acc), i))

            # print("The number of selected unlabeled data: {0} ".format(len(selected_pseudoLabel)))

            # print("The acc of the pseudolabel after seclected: {0} ".format(str(selected_acc)))

            # print("The number of selected pseudolabels in all classes : {0} ".format(selected_pseudolabel_list) )
            # print()
            
        # mini_batch
        for step in range(args.inner_iters):
            mini_data = torch.tensor([]).cuda()
            mini_label = torch.tensor([]).cuda()
            for j in range(args.way):
                num_mini_batch = min(selected_pseudolabel_list) 
                data_everyClass = torch.where(selected_pseudoLabel==j)[0]
                data_everyClass = data_everyClass.cpu().numpy()

                # if step == args.inner_iters-1:
                #     selected_entropy_sort = selected_U_pred[data_everyClass].argsort(descending=False)[:num_mini_batch]
                #     mini_data = torch.concat([mini_data, selected_unlabeld_data[data_everyClass][selected_entropy_sort]], 0)
                #     mini_label = torch.concat([mini_label, selected_pseudoLabel[data_everyClass][selected_entropy_sort]], 0)

                # else:
                mini_index = np.random.choice(data_everyClass, num_mini_batch, replace=False)  
                mini_data = torch.concat([mini_data, selected_unlabeld_data[mini_index]], 0)
                mini_label = torch.concat([mini_label, selected_pseudoLabel[mini_index]], 0)
                
            mix_x = torch.concat([support_data, mini_data], 0)
            mix_y = torch.concat([support_label, mini_label], 0)

            x1_feature, aug_pred1 = encoder(mix_x, updated_params_en, is_embedding = True)
            aug_pred2 = classier_aug(x1_feature.detach(), updated_params_cl)

            loss1 = F.cross_entropy(aug_pred1  , mix_y.long())
            loss2 = F.cross_entropy(aug_pred2  , mix_y.long())
        
            updated_params_en = update_params(loss1 , updated_params_en, step_size=args.gd_lr)
            updated_params_cl = update_params(loss2 , updated_params_cl, step_size=args.gd_lr*4)

            aug_s_and_u_feature, aug_s_and_u_pred1 = encoder(mix_x, updated_params_en, is_embedding = True)
            aug_s_and_u_pred2 = classier_aug(aug_s_and_u_feature.detach(), updated_params_cl)

            loss_kl1 = 0.4 * loss_KL(aug_s_and_u_pred1 , aug_s_and_u_pred2.detach())
            loss_kl2 = loss_KL(aug_s_and_u_pred2, aug_s_and_u_pred1.detach())
            updated_params_en = update_params(loss_kl1, updated_params_en, step_size=args.gd_lr)
            updated_params_cl = update_params(loss_kl2, updated_params_cl, step_size=args.gd_lr)
        with torch.no_grad():
            U_pred = encoder(unlabeled_data, updated_params_en)
            U_pred = torch.softmax(U_pred, dim=-1)
            unlabeled_pseudoLabel = torch.argmax(U_pred, dim=-1).detach()
            logitis_query = encoder(query, updated_params_en) / 0.5
            pesudo_acc = np.mean((unlabeled_pseudoLabel.cpu().numpy() == unlabeled_label.cpu().numpy()).tolist())
            acc = count_acc(logitis_query, query_labeld)
            with open('vis/data/our_pesudo_acc.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([pesudo_acc])
            with open('vis/data/our_test_acc.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([acc])
    print()

    return updated_params_en


class MAML_full(nn.Module):

    def __init__(self, args):
        super().__init__()
       
        if args.backbone_class == 'Res12':
            from model.networks.res12_maml import ResNet12_FSL_FULL
            self.encoder = ResNet12_FSL_FULL(args, args.way, dropblock_size=args.dropblock_size)
            self.classier_aug = Classifer(args, args.way)
        else:
            raise ValueError('')
        self.args = args

    def forward(self, data_shot, data_query, task_num):
        # update with gradient descent
        updated_params = inner_train_step_mine(self.encoder, self.classier_aug, data_shot, data_query, self.args, 'train', task_num)

        logitis = self.encoder(data_query.cuda(), updated_params) / self.args.temperature
        return logitis

    def forward_eval(self, data_shot, data_query, task_num):
        # update with gradient descent
        self.train()
        updated_params = inner_train_step_mine(self.encoder, self.classier_aug, data_shot, data_query, self.args, "val", task_num)

        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_query = self.encoder(data_query.cuda(), updated_params) / self.args.temperature
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