
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


def get_attacker_accuracy(model,new_attack):
	model.eval()

	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(device)
	acc_curve = []
	acc_total = np.ones(len(test_loader.dataset))

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
			pred_right = (pred==test_labels).nonzero().squeeze() 		
			acc_total[bstart:bend] = acc_total[bstart:bend] * (pred==test_labels).cpu().numpy()

			test_images = test_images[pred_right] 
			test_labels = test_labels[pred_right]

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
			adv_images, p = apply_attacker(test_images, attack_name, test_labels, model, attack_eps, previous_p, int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=gpu_idx,)
			pred = predict_from_logits(model(adv_images.detach()))
			acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy() 
			batch_idx += 1

			print('accuracy_total: {}/{}'.format(int(acc_total.sum()), len(test_loader.dataset)))
			print('natural_acc_oneshot: ', clean_acc_num/total_num)
			print('robust_acc_oneshot: ', (total_num-len(test_loader.dataset)+acc_total.sum()) /total_num)
		acc_curve.append(acc_total.sum())
		print('accuracy_curve: ', acc_curve)
	return acc_total

def get_policy_accuracy(model,policy):
    result_acc_total = np.ones(len(test_loader.dataset))

    for new_attack in policy:
        tmp_acc_total = get_attacker_accuracy(model,new_attack)
        
        result_acc_total = list(map(int,result_acc_total))
        tmp_acc_total = list(map(int,tmp_acc_total))

        result_acc_total = np.bitwise_and(result_acc_total,tmp_acc_total)

    return result_acc_total

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


args = parser.parse_args()
print(args)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')


if args.dataset == 'cifar10':
    args.num_classes = 10
    cifar10_val = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, transform = transforms.ToTensor(),download = True)
    test_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)

    if args.net_type == 'madry_adv_resnet50':
        #demo
        # from cifar_models.resnet import resnet50
        # model = resnet50()
        # model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_linf_8.pt')['state_dict'].items() if 'attacker' not in k and 'new' not in k})
        # normalize = NormalizeByChannelMeanStd(
        #     mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        # model = nn.Sequential(normalize, model)

        ###########################################################################
        # robust model-Linf-1
        # from robustbench.utils import load_model
        # print("robust Linf model-1")
        # model = load_model(model_name="Rebuffi2021Fixing_70_16_cutmix_extra",dataset='cifar10', threat_model='Linf')

        ###########################################################################
        # robust model-Linf-3
        # from robustbench.utils import load_model
        # print("robust Linf model-3")
        # model = load_model(model_name="Rebuffi2021Fixing_106_16_cutmix_ddpm",dataset='cifar10', threat_model='Linf')

        ###########################################################################
        # robust model-Linf-4
        # from robustbench.utils import load_model
        # print("robust Linf model-4")
        # model = load_model(model_name="Rebuffi2021Fixing_70_16_cutmix_ddpm",dataset='cifar10', threat_model='Linf')


        ###########################################################################
        # robust model-5
        # from robustbench.utils import load_model
        # print("robust Linf model-5")
        # model = load_model(model_name="Rade2021Helper_extra",dataset='cifar10', threat_model='Linf')


        ###########################################################################
        # robust model-6
        # from robustbench.utils import load_model
        # print("robust Linf model-6")
        # model = load_model(model_name="Gowal2020Uncovering_28_10_extra",dataset='cifar10', threat_model='Linf')


        ###########################################################################
        # robust model-Linf-7
        # from robustbench.utils import load_model
        # print("robust Linf model-7")
        # model = load_model(model_name="Rade2021Helper_ddpm",dataset='cifar10', threat_model='Linf')

        ###########################################################################
        # robust model-Linf-8
        from robustbench.utils import load_model
        print("robust Linf model-8")
        model = load_model(model_name="Rebuffi2021Fixing_28_10_cutmix_ddpm",dataset='cifar10', threat_model='Linf')

        ###########################################################################
        # robust model-Linf-10
        # from robustbench.utils import load_model
        # print("robust Linf model-10")
        # model = load_model(model_name="Sridhar2021Robust_34_15",dataset='cifar10', threat_model='Linf')

        ###########################################################################
        # robust model-Linf-12
        # from robustbench.utils import load_model
        # print("robust Linf model-12")
        # model = load_model(model_name="Sridhar2021Robust",dataset='cifar10', threat_model='Linf')

        
        ###########################################################################
        # robust model-13
        # from robustbench.utils import load_model
        # print("robust model-13")
        # model = load_model(model_name="Zhang2020Geometry",dataset='cifar10', threat_model='Linf')

        # ##########################################################################
        # # robust model-14
        # from robustbench.utils import load_model
        # print("robust model-14")
        # model = load_model(model_name="Carmon2019Unlabeled",dataset='cifar10', threat_model='Linf')

        # ##########################################################################
        # # robust model-15
        # from robustbench.utils import load_model
        # print("robust model-15")
        # model = load_model(model_name="Sehwag2021Proxy",dataset='cifar10', threat_model='Linf')

        # ##########################################################################
        # # robust linf model-16
        # from robustbench.utils import load_model
        # print("robust Linf model-16")
        # model = load_model(model_name="Rade2021Helper_R18_extra",dataset='cifar10', threat_model='Linf')

    elif args.net_type == 'madry_adv_resnet50_l2':

        # from cifar_models.resnet import resnet50
        # model = resnet50()
        # model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_l2_0_5.pt')['state_dict'].items() if 'attacker' not in k and 'new' not in k})
        # normalize = NormalizeByChannelMeanStd(
        #     mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        # model = nn.Sequential(normalize, model)

        ##########################################################################
        # # robust l2 model-8
        # from robustbench.utils import load_model
        # print("robust l2 model-8")
        # model = load_model(model_name="Sehwag2021Proxy",dataset='cifar10', threat_model='L2')

        ##########################################################################
        # # robust l2 model-10
        # from robustbench.utils import load_model
        # print("robust l2 model-10")
        # model = load_model(model_name="Gowal2020Uncovering",dataset='cifar10', threat_model='L2')

        ##########################################################################
        # # robust l2 model-12
        # from robustbench.utils import load_model
        # print("robust l2 model-12")
        # model = load_model(model_name="Sehwag2021Proxy_R18",dataset='cifar10', threat_model='L2')

        pass

    else:
        raise Exception('The net_type of {} is not supported by now!'.format(args.net_type))




#linf policy
policy = [{'attacker': 'ApgdDlrAttack', 'magnitude': 8/255, 'step': 32}, 
{'attacker': 'ApgdCeAttack', 'magnitude': 8/255, 'step': 32},
{'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 63},
{'attacker': 'FabAttack', 'magnitude': 8/255, 'step': 63}, 
{'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 126},
{'attacker': 'ApgdDlrAttack', 'magnitude': 8/255, 'step': 160},
{'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 378}]

#l2 policy
# policy = [{'attacker': 'DDNL2Attack', 'magnitude': None, 'step': 124}, 
# {'attacker': 'ApgdCeAttack', 'magnitude': 0.5, 'step': 31}, 
# {'attacker': 'DDNL2Attack', 'magnitude': None, 'step': 624}, 
# {'attacker': 'ApgdDlrAttack', 'magnitude': 0.5, 'step': 31}, 
# {'attacker': 'MultiTargetedAttack', 'magnitude': 0.5, 'step': 62}]

result_accuray = get_policy_accuracy(model,policy)
print(result_accuray.sum())