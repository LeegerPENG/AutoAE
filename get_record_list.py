import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np

import torchvision
import imageio

from torchvision import transforms
import argparse
from attack_ops import apply_attacker
from tqdm import tqdm
from tv_utils import ImageNet,Permute
import copy
import pickle
import random

gpu_idx = 0

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def record_get_attacker_accuracy(model,new_attack,copy_acc_total):
	model.eval()
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(device)
	acc_curve = []
	acc_total = copy.deepcopy(copy_acc_total)	
	total_record_list = []
	for i in range(new_attack['step']):
		total_record_list.append([])

	for _ in range(args.num_restarts):
		total_num = 0
		clean_acc_num = 0
		adv_acc_num = 0
		attack_successful_num = 0	
		batch_idx = 0
		for loaded_data in tqdm(test_loader):
			test_images, test_labels = loaded_data[0], loaded_data[1]
			bstart = batch_idx * args.batch_size
			if test_labels.size(0) < args.batch_size:
				bend = batch_idx * args.batch_size + test_labels.size(0)
			else:
				bend = (batch_idx+1) * args.batch_size
			test_images, test_labels = test_images.to(device), test_labels.to(device)
			total_num += test_labels.size(0)
			clean_logits = model(test_images)
			pred = predict_from_logits(clean_logits)
			acc_total[bstart:bend] = acc_total[bstart:bend] * (pred==test_labels).cpu().numpy()
			if len(test_images.shape) == 3:
				test_images = test_images.unsqueeze(0)
				test_labels = test_labels.unsqueeze(0)
			if len(test_labels.size()) == 0:
				clean_acc_num += 1
			else:
				clean_acc_num += test_labels.size(0)
			previous_p = None
			attack_name = new_attack['attacker']
			attack_eps = new_attack['magnitude']
			attack_steps = new_attack['step']
			adv_images, p, record_list= apply_attacker(test_images, attack_name, test_labels, model, attack_eps, previous_p, int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=gpu_idx,)
			pred = predict_from_logits(model(adv_images.detach()))
			acc_total[bstart:bend]= acc_total[bstart:bend] * (pred==test_labels).cpu().numpy() 
			batch_idx += 1
			print('accuracy_total: {}/{}'.format(int(acc_total.sum()), len(test_loader.dataset)))
			print('natural_acc_oneshot: ', clean_acc_num/total_num)
			print('robust_acc_oneshot: ', (total_num-len(test_loader.dataset)+acc_total.sum()) /total_num)
			total_record_list = np.hstack((total_record_list,record_list))
		acc_curve.append(acc_total.sum())
		with open("Record_list_"+attack_name+".pkl","wb") as f:
			pickle.dump(total_record_list, f)
	return acc_total

def record_append_next_attack(model,last_acc_total,t_max):
	max_result = 0
	original_accuracy = last_acc_total.sum()
	best_attacker = None
	best_acc_total = None
	best_t = None
	acc_total = copy.deepcopy(last_acc_total)
	for attack_idx in range(len(candidate_pool)):
		new_attack = candidate_pool[attack_idx]
		for t in range(1000,t_max+1,125):
			new_attack['step']= t 
			tmp_acc_total = record_get_attacker_accuracy(model,new_attack,acc_total)
			cur_result = abs(original_accuracy-tmp_acc_total.sum())/t
			if cur_result>max_result:
				best_t = copy.deepcopy(t)
				best_acc_total = copy.deepcopy(tmp_acc_total)
				best_attacker = copy.deepcopy(new_attack)
	return [best_attacker,best_acc_total,best_t]

def record_greedy_algorithm(model):
	policy = []
	acc_total = np.ones(len(test_loader.dataset))
	t_max = 1000
	while t_max > 0:
		[next_attack, acc_total, t] = record_append_next_attack(model, acc_total, t_max)
		if next_attack is None:
			return policy
		policy.append(next_attack)
		t_max = t_max - t
	return policy


def get_5000_train_data(cifar10_train):
    selected_cifar_train = []
    for i in range(10):
        sub_selected_cifar_train = []
        for item in cifar10_train:
            if item[1]==i:
                sub_selected_cifar_train.append(item)
            if len(sub_selected_cifar_train)==500:
                break
        selected_cifar_train.extend(sub_selected_cifar_train)
    return selected_cifar_train

parser = argparse.ArgumentParser(description='Random search of Auto-attack')
parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn | ile')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='madry_adv_resnet50', help='resnet18 | resnet50 | inception_v3 | densenet121 | vgg16_bn')
parser.add_argument('--num_restarts', type=int, default=1, help='the # of classes')
parser.add_argument('--max_epsilon', type=float, default=8/255, help='the attack sequence length')
parser.add_argument('--ensemble', action='store_true', help='the attack sequence length')
parser.add_argument('--transfer_test', action='store_true', help='the attack sequence length')
parser.add_argument('--sub_net_type', default='madry_adv_resnet50', help='resnet18 | resnet50 | inception_v3 | densenet121 | vgg16_bn')
parser.add_argument('--target', action='store_true', default=False)
parser.add_argument('--norm', default='linf', help='linf | l2 | unrestricted')
parser.add_argument('--linf_attacker', default='Record_CWAttack_adaptive_stepsize_Linf', help='RecordFabAttack_Linf | RecordApgdDlrAttack_Linf | RecordApgdCeAttack_Linf |\
Record_CWAttack_adaptive_stepsize_Linf | RecordMultiTargetedAttack_Linf ')
parser.add_argument('--l2_attacker', default='Record_CWAttack_adaptive_stepsize_L2', help='Record_CWAttack_adaptive_stepsize_L2 | RecordMultiTargetedAttack_L2 | RecordApgdCeAttack_L2 |\
RecordApgdDlrAttack_L2 | RecordFabAttack_L2 | Record_PGD_Attack_adaptive_stepsize_L2 | RecordDDNL2Attack_L2')

args = parser.parse_args()
print(args)

RecordMultiTargetedAttack_L2 = {'attacker': 'RecordMultiTargetedAttack', 'magnitude': 0.5, 'step': 50}
RecordMultiTargetedAttack_Linf = {'attacker': 'RecordMultiTargetedAttack', 'magnitude': 8/255, 'step': 50}
Record_CWAttack_adaptive_stepsize_L2 = {'attacker': 'Record_CW_Attack_adaptive_stepsize', 'magnitude': 0.5, 'step': 50}
Record_CWAttack_adaptive_stepsize_Linf = {'attacker': 'Record_CW_Attack_adaptive_stepsize', 'magnitude': 8/255, 'step': 50}
Record_PGD_Attack_adaptive_stepsize_L2 = {'attacker': 'Record_PGD_Attack_adaptive_stepsize', 'magnitude': 0.5, 'step': 50}
RecordDDNL2Attack_L2 = {'attacker': 'RecordDDNL2Attack', 'magnitude': None, 'step': 50}
RecordApgdCeAttack_L2 = {'attacker': 'RecordApgdCeAttack', 'magnitude': 0.5, 'step': 50}
RecordApgdCeAttack_Linf = {'attacker': 'RecordApgdCeAttack', 'magnitude': 8/255, 'step': 50}
RecordApgdDlrAttack_L2 ={'attacker': 'RecordApgdDlrAttack', 'magnitude': 0.5, 'step': 50}
RecordApgdDlrAttack_Linf =  {'attacker': 'RecordApgdDlrAttack', 'magnitude': 8/255, 'step': 50}
RecordFabAttack_L2 = {'attacker': 'RecordFabAttack', 'magnitude': 0.5, 'step': 50}
RecordFabAttack_Linf = {'attacker': 'RecordFabAttack', 'magnitude': 8/255, 'step': 50}


if args.norm == 'linf':
	if args.linf_attacker == 'RecordFabAttack_Linf':
		candidate_pool = [RecordFabAttack_Linf]
	elif args.linf_attacker == 'RecordApgdDlrAttack_Linf':
		candidate_pool = [RecordApgdDlrAttack_Linf]
	elif args.linf_attacker == 'RecordApgdCeAttack_Linf':
		candidate_pool = [RecordApgdCeAttack_Linf]
	elif args.linf_attacker == 'Record_CWAttack_adaptive_stepsize_Linf':
		candidate_pool = [Record_CWAttack_adaptive_stepsize_Linf]
	elif args.linf_attacker == 'RecordMultiTargetedAttack_Linf':
		candidate_pool = [RecordMultiTargetedAttack_Linf]

elif args.norm == 'l2':
	if args.l2_attacker =='RecordDDNL2Attack_L2':
		candidate_pool = [RecordDDNL2Attack_L2]
	elif args.l2_attacker =='Record_PGD_Attack_adaptive_stepsize_L2':
		candidate_pool = [Record_PGD_Attack_adaptive_stepsize_L2]
	elif args.l2_attacker =='RecordFabAttack_L2':
		candidate_pool = [RecordFabAttack_L2]
	elif args.l2_attacker =='RecordApgdDlrAttack_L2':
		candidate_pool = [RecordApgdDlrAttack_L2]
	elif args.l2_attacker =='RecordApgdCeAttack_L2':
		candidate_pool = [RecordApgdCeAttack_L2]
	elif args.l2_attacker =='RecordMultiTargetedAttack_L2':
		candidate_pool = [RecordMultiTargetedAttack_L2]
	elif args.l2_attacker =='Record_CWAttack_adaptive_stepsize_L2':
		candidate_pool = [Record_CWAttack_adaptive_stepsize_L2]

print('candidate_pool: ', candidate_pool)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

if args.dataset == 'cifar10':
    args.num_classes = 10

    cifar10_train = torchvision.datasets.CIFAR10(root='/root/project/data/cifar10', train=True, transform = transforms.ToTensor())
    selected_cifar_train = get_5000_train_data(cifar10_train)
    test_loader = torch.utils.data.DataLoader(selected_cifar_train, batch_size=args.batch_size,shuffle=False, pin_memory=True, num_workers=8)
    if args.net_type == 'madry_adv_resnet50':
        from cifar_models.resnet import resnet50
        model = resnet50()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_linf_8.pt')['state_dict'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model)

    elif args.net_type == 'madry_adv_resnet50_l2':
        from cifar_models.resnet import resnet50
        model = resnet50()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_l2_0_5.pt')['state_dict'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model)
    else:
        raise Exception('The net_type of {} is not supported by now!'.format(args.net_type))
result_policy = record_greedy_algorithm(model)
