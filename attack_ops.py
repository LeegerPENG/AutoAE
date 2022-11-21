import numpy as np
from itertools import product, repeat
import PIL
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from tv_utils import SpatialAffine, GaussianSmoothing
from attack_utils import projection_linf, check_shape, dlr_loss, get_diff_logits_grads_batch
from imagenet_c import corrupt
import torch.optim as optim
import math
from advertorch.attacks import LinfSPSAAttack
import os
import time


from fab_projections import fab_projection_linf, projection_l2
#from torch.autograd.gradcheck import zero_gradients  #for torch 1.9




def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def check_oscillation(x, j, k, y5, k3=0.5):
    t = np.zeros(x.shape[1])
    for counter5 in range(k):
        t += x[j - counter5] > x[j - counter5 - 1]
    return t <= k*k3*np.ones(t.shape)

def CW_Attack_adaptive_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    
    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    # print(x.shape)

    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    one_hot_y = torch.zeros(y.size(0), 10).to(device)
    one_hot_y[torch.arange(y.size(0)), y] = 1
    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        correct_logit = torch.sum(one_hot_y * logits, dim=1)
        wrong_logit,_ = torch.max((1-one_hot_y) * logits-1e4*one_hot_y, dim=1)

        loss_indiv = -F.relu(correct_logit-wrong_logit+50)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            correct_logit = torch.sum(one_hot_y * logits, dim=1)
            wrong_logit,_ = torch.max((1-one_hot_y) * logits-1e4*one_hot_y, dim=1)

            loss_indiv = -F.relu(correct_logit-wrong_logit+50)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv[ind_non_suc] = x_best_adv
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def Record_CW_Attack_adaptive_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    
    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()

    ind_suc = (pred!=y).nonzero().squeeze()
    record_list = []

    
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)

    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    one_hot_y = torch.zeros(y.size(0), 10).to(device)
    one_hot_y[torch.arange(y.size(0)), y] = 1
    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        correct_logit = torch.sum(one_hot_y * logits, dim=1)
        wrong_logit,_ = torch.max((1-one_hot_y) * logits-1e4*one_hot_y, dim=1)

        loss_indiv = -F.relu(correct_logit-wrong_logit+50)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1) 
            pred_after_attack = predict_from_logits(logits)

            record = np.ones(len(pred_after_attack))
            record = record * (pred_after_attack==y).cpu().numpy() 
            record_list.append(record)

            correct_logit = torch.sum(one_hot_y * logits, dim=1)
            wrong_logit,_ = torch.max((1-one_hot_y) * logits-1e4*one_hot_y, dim=1)

            loss_indiv = -F.relu(correct_logit-wrong_logit+50)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)
    adv[ind_non_suc] = x_best_adv[ind_non_suc]
    
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    for item in record_list:
        item[ind_suc.cpu().numpy()]=0

    return adv, now_p, record_list

def MultiTargetedAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv_out = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
    
    def run_once(model, x_in, y_in, magnitude, max_iters, _type, target_class, max_eps, previous_p):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        # print(x.shape)
        if previous_p is not None:
            max_x = x - previous_p + max_eps
            min_x = x - previous_p - max_eps
        else:
            max_x = x + max_eps
            min_x = x - max_eps

        n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
        if _type == 'linf':
            t = 2 * torch.rand(x.shape).to(device).detach() - 1
            x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
            x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
        elif _type == 'l2':
            t = torch.randn(x.shape).to(device).detach()
            x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
            if previous_p is not None:
                x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([max_iters, x.shape[0]])
        loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        output = model(x)
        y_target = output.sort(dim=1)[1][:, -target_class]
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(1):
            with torch.enable_grad():
                logits = model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv = dlr_loss(logits, y, y_target)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        
        grad_best = grad.clone()
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(max_iters):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                if _type == 'linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - magnitude), x + magnitude), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                    
                elif _type == 'l2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                    if previous_p is not None:
                        x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                            max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                   
                x_adv = x_adv_1 + 0.
            
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(1):
                with torch.enable_grad():
                    logits = model(x_adv) # 1 forward pass (eot_iter = 1)
                    loss_indiv = dlr_loss(logits, y, y_target)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - size_decr, n_iter_min)

        return acc, x_best_adv


    adv = x.clone()
    for target_class in range(2, 9 + 2):
        acc_curr, adv_curr = run_once(model, x, y, magnitude, max_iters, _type, target_class, max_eps, previous_p)
        ind_curr = (acc_curr == 0).nonzero().squeeze()
        adv[ind_curr] = adv_curr[ind_curr].clone()

    now_p = adv-x
    adv_out[ind_non_suc] = adv

    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv_out, previous_p_c

    return adv_out, now_p


def RecordMultiTargetedAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv_out = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv_out, previous_p

    ind_non_suc = (pred==y).nonzero().squeeze()
    ind_suc = (pred!=y).nonzero().squeeze()


    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
    
    def run_once(model, x_in, y_in, magnitude, max_iters, _type, target_class, max_eps, previous_p):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        # print(x.shape)
        if previous_p is not None:
            max_x = x - previous_p + max_eps
            min_x = x - previous_p - max_eps
        else:
            max_x = x + max_eps
            min_x = x - max_eps

        n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
        if _type == 'linf':
            t = 2 * torch.rand(x.shape).to(device).detach() - 1
            x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
            x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
        elif _type == 'l2':
            t = torch.randn(x.shape).to(device).detach()
            x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
            if previous_p is not None:
                x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([max_iters, x.shape[0]])
        loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        output = model(x)
        y_target = output.sort(dim=1)[1][:, -target_class]
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(1):
            with torch.enable_grad():
                logits = model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv = dlr_loss(logits, y, y_target)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        
        grad_best = grad.clone()
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(max_iters):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                if _type == 'linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - magnitude), x + magnitude), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                    
                elif _type == 'l2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                    if previous_p is not None:
                        x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                            max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                   
                x_adv = x_adv_1 + 0.
            
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(1):
                with torch.enable_grad():
                    logits = model(x_adv) # 1 forward pass (eot_iter = 1)
                    pred_after_attack = predict_from_logits(logits)
                    record = np.ones(len(pred_after_attack))
                    record = record * (pred_after_attack==y).cpu().numpy() 
                    record_list.append(record)
                    loss_indiv = dlr_loss(logits, y, y_target)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - size_decr, n_iter_min)
            for item in record_list:
                item[ind_suc.cpu().numpy()]=0
                #item[ind_suc]=0
        return acc, x_best_adv,record_list


    adv = x.clone()
    result_record_list =[]
    for target_class in range(2, 9 + 2): 
        record_list = []
        acc_curr, adv_curr,new_record_list= run_once(model, x, y, magnitude, max_iters, _type, target_class, max_eps, previous_p)

        if len(result_record_list)==0:
            result_record_list = new_record_list
        else:

            for i in range(len(result_record_list)):
                for j in range(len(result_record_list[i])):

                    result_record_list[i][j] = int(result_record_list[i][j])&int(new_record_list[i][j])

        ind_curr = (acc_curr == 0).nonzero().squeeze()

        adv[ind_curr] = adv_curr[ind_curr].clone()

    now_p = adv-x
    adv_out[ind_non_suc]= adv[ind_non_suc]
    #adv_out[ind_non_suc] = adv

    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv_out, previous_p_c

    return adv_out, now_p,result_record_list




def ApgdCeAttack(x,y,model,magnitude,previous_p,max_eps,max_iters=20,target=None,_type='l2',gpu_idx=None,seed=time.time(),n_restarts=1):

    device = 'cuda:{}'.format(gpu_idx)

    if not y is None and len(y.shape) == 0:
        x.unsqueeze_(0)
        y.unsqueeze_(0)


    x = x.detach().clone().float().to(device)
    
    y_pred = model(x).max(1)[1]

    if y is None:
        y = y_pred.detach().clone().long().to(device)
    else:
        y = y.detach().clone().long().to(device)

    adv = x.clone()

    acc = y_pred == y
    loss = -1e10 * torch.ones_like(acc).float()

    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)

    for counter in range(n_restarts):
        ind_to_fool = acc.nonzero().squeeze()
        if len(ind_to_fool.shape) == 0:
            ind_to_fool = ind_to_fool.unsqueeze(0)
        if ind_to_fool.numel() != 0: 
            x_to_fool = x[ind_to_fool].clone()
            y_to_fool = y[ind_to_fool].clone()
            res_curr = apgd_attack_single_run(x= x_to_fool, y = y_to_fool,
                model = model,_type=_type,gpu_idx=gpu_idx,n_iter=max_iters,loss='ce',eps = magnitude)       

            best_curr, acc_curr, loss_curr, adv_curr = res_curr
            ind_curr = (acc_curr == 0).nonzero().squeeze()
            acc[ind_to_fool[ind_curr]] = 0
            adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

    return adv,None

def apgd_attack_single_run(x, y,model,eps,x_init=None,_type=None,gpu_idx=None,n_iter=100,loss='ce',eot_iter=20,thr_decr=.75):

    device = 'cuda:{}'.format(gpu_idx)

    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)

    if len(x.shape) < ndims:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x + eps * torch.ones_like(x).detach() * normalize(t,_type)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x + eps * torch.ones_like(x).detach() * normalize(t,_type)

    if not x_init is None:
        x_adv = x_init.clone()

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]]).to(device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]]).to(device)
    acc_steps = torch.zeros_like(loss_best_steps)

    if loss == 'ce':
        criterion_indiv = nn.CrossEntropyLoss(reduction='none')
    elif loss == 'dlr':
        criterion_indiv = auto_attack_dlr_loss
    else:
        raise ValueError('unknowkn loss')

    x_adv.requires_grad_()
    grad = torch.zeros_like(x)
    for _ in range(eot_iter):

        with torch.enable_grad():
            logits = model(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()

        grad += torch.autograd.grad(loss, [x_adv])[0].detach()

    
    grad /= float(eot_iter)
    grad_best = grad.clone()

    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    alpha = 2. if _type in ['linf', 'l2'] else 1. if _type in ['L1'] else 2e-2

    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)
    


    step_size = alpha * eps * torch.ones([x.shape[0], *([1] * ndims)]).to(device).detach()
    x_adv_old = x_adv.clone()
    counter = 0

    n_iter_2 = max(int(0.22 * n_iter), 1)
    k = n_iter_2 + 0

    counter3 = 0

    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0

    n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
    u = torch.arange(x.shape[0], device=device)
    for i in range(n_iter):
        ### gradient step
        with torch.no_grad():
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()

            a = 0.75 if i > 0 else 1.0

            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                    x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - eps), x + eps), 0.0, 1.0)

            elif _type == 'l2':
                x_adv_1 = x_adv + step_size * normalize(grad, _type)
                x_adv_1 = torch.clamp(x + normalize(x_adv_1 - x,_type
                    ) * torch.min(eps * torch.ones_like(x).detach(),
                    lp_norm(x_adv_1 - x,_type=_type)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(x + normalize(x_adv_1 - x, _type
                    ) * torch.min(eps * torch.ones_like(x).detach(),
                    lp_norm(x_adv_1 - x,_type=_type)), 0.0, 1.0)  
            x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(eot_iter):
            with torch.enable_grad():
                logits = model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        
        grad /= float(eot_iter)

        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = (pred == 0).nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.

        # if self.verbose:
        #     str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
        #         step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
        #     print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
        #         i, loss_best.sum(), acc.float().mean(), str_stats))

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1 + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0

            counter3 += 1

            if counter3 == k:
                if _type in ['linf', 'l2']:
                    fl_oscillation = apgd_check_oscillation(loss_steps, i, k,
                        loss_best, k3=thr_decr,gpu_idx=gpu_idx)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                        loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation,
                        fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        n_reduced = fl_oscillation.sum()

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    size_decr = max(int(0.03 * n_iter), 1)
                    n_iter_min = max(int(0.06 * n_iter), 1)
                    k = max(k - size_decr, n_iter_min)   
                counter3 = 0

    return (x_best, acc, loss_best, x_best_adv)


def RecordApgdCeAttack(x,y,model,magnitude,previous_p,max_eps,max_iters=20,target=None,_type='l2',gpu_idx=None,seed=time.time(),n_restarts=1):

    device = 'cuda:{}'.format(gpu_idx)

    if not y is None and len(y.shape) == 0:
        x.unsqueeze_(0)
        y.unsqueeze_(0)


    x = x.detach().clone().float().to(device)
    
    y_pred = model(x).max(1)[1]

    if y is None:
        y = y_pred.detach().clone().long().to(device)
    else:
        y = y.detach().clone().long().to(device)

    adv = x.clone()
    acc = y_pred == y 
    ori_non_acc = y_pred != y 
    loss = -1e10 * torch.ones_like(acc).float()
    for counter in range(n_restarts):
        x_to_fool = x.clone()
        y_to_fool = y.clone()
        res_curr = record_apgd_attack_single_run(x= x_to_fool, y = y_to_fool,
            model = model,_type=_type,gpu_idx=gpu_idx,n_iter=max_iters,loss='ce',eps = magnitude)       

        best_curr, acc_curr, loss_curr, adv_curr,record_list = res_curr
        ind_curr = (acc_curr == 0).nonzero().squeeze()

        acc[ind_curr] = 0
        adv[ind_curr] = adv_curr[ind_curr].clone()
    ind_not_to_fool = ori_non_acc.nonzero().squeeze()


    for item in record_list:
        item[ind_not_to_fool.cpu().numpy()] = 0

    return adv,None,record_list

def record_apgd_attack_single_run(x, y,model,eps,x_init=None,_type=None,gpu_idx=None,n_iter=100,loss='ce',eot_iter=20,thr_decr=.75):

    device = 'cuda:{}'.format(gpu_idx)

    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)

    if len(x.shape) < ndims:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x + eps * torch.ones_like(x).detach() * normalize(t,_type)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x + eps * torch.ones_like(x).detach() * normalize(t,_type)


    if not x_init is None:
        x_adv = x_init.clone()

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]]).to(device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]]).to(device)
    acc_steps = torch.zeros_like(loss_best_steps)

    if loss == 'ce':
        criterion_indiv = nn.CrossEntropyLoss(reduction='none')
    elif loss == 'dlr':
        criterion_indiv = auto_attack_dlr_loss
    else:
        raise ValueError('unknowkn loss')

    x_adv.requires_grad_()
    grad = torch.zeros_like(x)
    for _ in range(eot_iter):

        with torch.enable_grad():
            logits = model(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()

        grad += torch.autograd.grad(loss, [x_adv])[0].detach()

    
    grad /= float(eot_iter)
    grad_best = grad.clone()

    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    alpha = 2. if _type in ['linf', 'l2'] else 1. if _type in ['L1'] else 2e-2

    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)
    


    step_size = alpha * eps * torch.ones([x.shape[0], *([1] * ndims)]).to(device).detach()
    x_adv_old = x_adv.clone()
    counter = 0

    n_iter_2 = max(int(0.22 * n_iter), 1)
    k = n_iter_2 + 0

    counter3 = 0

    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0

    n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
    u = torch.arange(x.shape[0], device=device)

    record_list = []

    for i in range(n_iter):
        ### gradient step
        with torch.no_grad():
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()

            a = 0.75 if i > 0 else 1.0

            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                    x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - eps), x + eps), 0.0, 1.0)

            elif _type == 'l2':
                x_adv_1 = x_adv + step_size * normalize(grad, _type)
                x_adv_1 = torch.clamp(x + normalize(x_adv_1 - x,_type
                    ) * torch.min(eps * torch.ones_like(x).detach(),
                    lp_norm(x_adv_1 - x,_type=_type)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(x + normalize(x_adv_1 - x, _type
                    ) * torch.min(eps * torch.ones_like(x).detach(),
                    lp_norm(x_adv_1 - x,_type=_type)), 0.0, 1.0)  
            x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(eot_iter):
            with torch.enable_grad():
                logits = model(x_adv)  

                loss_indiv = criterion_indiv(logits, y) #loss function
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        

        pred_after_attack = predict_from_logits(logits)
        record = np.ones(len(pred_after_attack))
        record= record * (pred_after_attack==y).cpu().numpy() 

        record_list.append(record)

        
        grad /= float(eot_iter)

        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = (pred == 0).nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.

        # if self.verbose:
        #     str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
        #         step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
        #     print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
        #         i, loss_best.sum(), acc.float().mean(), str_stats))

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1 + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0

            counter3 += 1

            if counter3 == k:
                if _type in ['linf', 'l2']:
                    fl_oscillation = apgd_check_oscillation(loss_steps, i, k,
                        loss_best, k3=thr_decr,gpu_idx=gpu_idx)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                        loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation,
                        fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        n_reduced = fl_oscillation.sum()

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    size_decr = max(int(0.03 * n_iter), 1)
                    n_iter_min = max(int(0.06 * n_iter), 1)
                    k = max(k - size_decr, n_iter_min)   
                counter3 = 0

    return (x_best, acc, loss_best, x_best_adv,record_list)

def normalize(x, _type):
    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)
    if _type == 'linf':
        t = x.abs().view(x.shape[0], -1).max(1)[0]
        return x / (t.view(-1, *([1] * ndims)) + 1e-12)

    elif _type == 'l2':
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return x / (t.view(-1, *([1] * ndims)) + 1e-12)

def auto_attack_dlr_loss(x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])
        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

def lp_norm(x,_type):
    if _type == 'l2':
        orig_dim = list(x.shape[1:])
        ndims = len(orig_dim)
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return t.view(-1, *([1] * ndims))

def apgd_check_oscillation(x, j, k, y5, k3=0.75,gpu_idx=None):
    device = 'cuda:{}'.format(gpu_idx)
    t = torch.zeros(x.shape[1]).to(device)
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()

def ApgdDlrAttack(x,y,model,magnitude,previous_p,max_eps,max_iters=20,target=None,_type='l2',gpu_idx=None,seed=time.time(),n_restarts=1):
    device = 'cuda:{}'.format(gpu_idx)

    if not y is None and len(y.shape) == 0:
        x.unsqueeze_(0)
        y.unsqueeze_(0)


    x = x.detach().clone().float().to(device)
    
    y_pred = model(x).max(1)[1]

    if y is None:
        y = y_pred.detach().clone().long().to(device)
    else:
        y = y.detach().clone().long().to(device)

    adv = x.clone()

    acc = y_pred == y
    loss = -1e10 * torch.ones_like(acc).float()

    startt = time.time()

    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)

    for counter in range(n_restarts):
        ind_to_fool = acc.nonzero().squeeze()
        if len(ind_to_fool.shape) == 0:
            ind_to_fool = ind_to_fool.unsqueeze(0)
        if ind_to_fool.numel() != 0:
            x_to_fool = x[ind_to_fool].clone()
            y_to_fool = y[ind_to_fool].clone()
            
            res_curr = apgd_attack_single_run(x= x_to_fool, y = y_to_fool,
                model = model,_type=_type,gpu_idx=gpu_idx,loss='dlr',eps = magnitude,n_iter=max_iters)       

            best_curr, acc_curr, loss_curr, adv_curr = res_curr
            ind_curr = (acc_curr == 0).nonzero().squeeze()
            acc[ind_to_fool[ind_curr]] = 0
            adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

    return adv,None

def RecordApgdDlrAttack(x,y,model,magnitude,previous_p,max_eps,max_iters=20,target=None,_type='l2',gpu_idx=None,seed=time.time(),n_restarts=1):

    device = 'cuda:{}'.format(gpu_idx)

    if not y is None and len(y.shape) == 0:
        x.unsqueeze_(0)
        y.unsqueeze_(0)


    x = x.detach().clone().float().to(device)
    
    y_pred = model(x).max(1)[1]

    if y is None:
        y = y_pred.detach().clone().long().to(device)
    else:
        y = y.detach().clone().long().to(device)

    adv = x.clone()

    acc = y_pred == y

    ori_non_acc = y_pred != y 


    loss = -1e10 * torch.ones_like(acc).float()



    for counter in range(n_restarts):


        x_to_fool = x.clone()
        y_to_fool = y.clone()

        res_curr = record_apgd_attack_single_run(x= x_to_fool, y = y_to_fool,
            model = model,_type=_type,gpu_idx=gpu_idx,n_iter=max_iters,loss='dlr',eps = magnitude)       

        best_curr, acc_curr, loss_curr, adv_curr,record_list = res_curr
        ind_curr = (acc_curr == 0).nonzero().squeeze()

        acc[ind_curr] = 0
        adv[ind_curr] = adv_curr[ind_curr].clone()
    

    ind_not_to_fool = ori_non_acc.nonzero().squeeze()


    for item in record_list:
        item[ind_not_to_fool.cpu().numpy()] = 0

    return adv,None,record_list

def FabAttack(x,y,model,magnitude,previous_p,max_eps,max_iters=20,target=None,_type='l2',gpu_idx=None,seed=time.time(),n_restarts=1):
    device = 'cuda:{}'.format(gpu_idx)

    #self.device = x.device
    adv = x.clone()
    with torch.no_grad():
        acc = model(x).max(1)[1] == y

        startt = time.time()

        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)

        for counter in range(n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                adv_curr = fab_attack_single_run(x_to_fool, y_to_fool,model=model,magnitude=magnitude,_type=_type,gpu_idx=gpu_idx,n_iter = max_iters,use_rand_start=(counter > 0), is_targeted=False)

                acc_curr = model(adv_curr).max(1)[1] == y_to_fool
                if _type == 'linf':
                    res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).max(1)[0]
                elif _type == 'l2':
                    res = ((x_to_fool - adv_curr) ** 2).reshape(x_to_fool.shape[0], -1).sum(dim=-1).sqrt()

                acc_curr = torch.max(acc_curr, res > magnitude)

                ind_curr = (acc_curr == 0).nonzero().squeeze()
                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
    return adv,None

def fab_attack_single_run(x, y,model,magnitude,_type,use_rand_start=False, is_targeted=False,gpu_idx=None,n_iter=100,alpha_max=0.1,eta=1.05,beta=0.9):
    """
    :param x:             clean images
    :param y:             clean labels, if None we use the predicted labels
    :param is_targeted    True if we ise targeted version. Targeted class is assigned by `self.target_class`
    """
    device = 'cuda:{}'.format(gpu_idx)

    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)

    x = x.detach().clone().float().to(device)

    y_pred = fab_get_predicted_label(x,model) 
    if y is None:
        y = y_pred.detach().clone().long().to(device)
    else:
        y = y.detach().clone().long().to(device)
    pred = y_pred == y
    corr_classified = pred.float().sum()
    # if self.verbose:
    #     print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
    if pred.sum() == 0:
        return x
    pred = fab_check_shape(pred.nonzero().squeeze())

    startt = time.time()
    # runs the attack only on correctly classified points
    im2 = x[pred].detach().clone()
    la2 = y[pred].detach().clone() 
    if len(im2.shape) == ndims:
        im2 = im2.unsqueeze(0)
    bs = im2.shape[0]
    u1 = torch.arange(bs)
    adv = im2.clone()
    adv_c = x.clone()
    res2 = 1e10 * torch.ones([bs]).to(device)
    res_c = torch.zeros([x.shape[0]]).to(device)
    x1 = im2.clone()
    x0 = im2.clone().reshape([bs, -1])
    counter_restarts = 0

    while counter_restarts < 1:
        if use_rand_start:
            if _type == 'linf':
                t = 2 * torch.rand(x1.shape).to(device) - 1
                x1 = im2 + (torch.min(res2,
                                        magnitude * torch.ones(res2.shape)
                                        .to(device)
                                        ).reshape([-1, *[1]*ndims])
                            ) * t / (t.reshape([t.shape[0], -1]).abs()
                                        .max(dim=1, keepdim=True)[0]
                                        .reshape([-1, *[1]*ndims])) * .5
            elif _type == 'l2':
                t = torch.randn(x1.shape).to(device)
                x1 = im2 + (torch.min(res2,
                                        magnitude * torch.ones(res2.shape)
                                        .to(device)
                                        ).reshape([-1, *[1]*ndims])
                            ) * t / ((t ** 2)
                                        .view(t.shape[0], -1)
                                        .sum(dim=-1)
                                        .sqrt()
                                        .view(t.shape[0], *[1]*ndims)) * .5

            x1 = x1.clamp(0.0, 1.0)

        counter_iter = 0
        while counter_iter < n_iter: 
            with torch.no_grad():
                df, dg = fab_get_diff_logits_grads_batch(x1, la2,model,gpu_idx)
                if _type == 'linf':
                    dist1 = df.abs() / (1e-12 +
                                        dg.abs()
                                        .reshape(dg.shape[0], dg.shape[1], -1)
                                        .sum(dim=-1))
                elif _type == 'l2':
                    dist1 = df.abs() / (1e-12 + (dg ** 2)
                                        .reshape(dg.shape[0], dg.shape[1], -1)
                                        .sum(dim=-1).sqrt())

                else:
                    raise ValueError('norm not supported')
                ind = dist1.min(dim=1)[1]
                dg2 = dg[u1, ind]
                b = (- df[u1, ind] + (dg2 * x1).reshape(x1.shape[0], -1)
                                        .sum(dim=-1))
                w = dg2.reshape([bs, -1])

                if _type == 'linf':
                    d3 = fab_projection_linf(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0))
                elif _type == 'l2':
                    d3 = projection_l2(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0))

                d1 = torch.reshape(d3[:bs], x1.shape)
                d2 = torch.reshape(d3[-bs:], x1.shape)
                if _type == 'linf':
                    a0 = d3.abs().max(dim=1, keepdim=True)[0]\
                        .view(-1, *[1]*ndims)
                elif _type == 'l2':
                    a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()\
                        .view(-1, *[1]*ndims)

                a0 = torch.max(a0, 1e-8 * torch.ones(
                    a0.shape).to(device))
                a1 = a0[:bs]
                a2 = a0[-bs:]
                alpha = torch.min(torch.max(a1 / (a1 + a2),
                                            torch.zeros(a1.shape)
                                            .to(device)),
                                    alpha_max * torch.ones(a1.shape)
                                    .to(device))
                x1 = ((x1 + eta * d1) * (1 - alpha) +
                        (im2 + d2 * eta) * alpha).clamp(0.0, 1.0)

                is_adv = fab_get_predicted_label(x1,model) != la2 


                if is_adv.sum() > 0:
                    ind_adv = is_adv.nonzero().squeeze()
                    ind_adv = fab_check_shape(ind_adv)
                    if _type == 'linf':
                        t = (x1[ind_adv] - im2[ind_adv]).reshape(
                            [ind_adv.shape[0], -1]).abs().max(dim=1)[0]
                    elif _type == 'l2':
                        t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                            .reshape(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                    
                    adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                        float().reshape([-1, *[1]*ndims]) + adv[ind_adv]\
                        * (t >= res2[ind_adv]).float().reshape(
                        [-1, *[1]*ndims]) 
                    res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                        + res2[ind_adv] * (t >= res2[ind_adv]).float()
                    x1[ind_adv] = im2[ind_adv] + (
                        x1[ind_adv] - im2[ind_adv]) * beta

                counter_iter += 1

        counter_restarts += 1

    ind_succ = res2 < 1e10
    # if self.verbose:
    #     print('success rate: {:.0f}/{:.0f}'
    #           .format(ind_succ.float().sum(), corr_classified) +
    #           ' (on correctly classified points) in {:.1f} s'
    #           .format(time.time() - startt))

    res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
    ind_succ = fab_check_shape(ind_succ.nonzero().squeeze())
    adv_c[pred[ind_succ]] = adv[ind_succ].clone()

    return adv_c


def RecordFabAttack(x,y,model,magnitude,previous_p,max_eps,max_iters=20,target=None,_type='l2',gpu_idx=None,seed=time.time(),n_restarts=1):
    device = 'cuda:{}'.format(gpu_idx)

    #self.device = x.device
    adv = x.clone()
    with torch.no_grad():
        acc = model(x).max(1)[1] == y

        startt = time.time()

        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)

        for counter in range(n_restarts):
            # ind_to_fool = acc.nonzero().squeeze()
            # if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
            # if ind_to_fool.numel() != 0:
            # x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()

            x_to_fool = x.clone()
            y_to_fool = y.clone()
            adv_curr,record_list = record_fab_attack_single_run(x_to_fool, y_to_fool,model=model,magnitude=magnitude,_type=_type,gpu_idx=gpu_idx,n_iter = max_iters,use_rand_start=(counter > 0), is_targeted=False)

            acc_curr = model(adv_curr).max(1)[1] == y_to_fool
            if _type == 'linf':
                res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).max(1)[0]
            elif _type == 'l2':
                res = ((x_to_fool - adv_curr) ** 2).reshape(x_to_fool.shape[0], -1).sum(dim=-1).sqrt()

            acc_curr = torch.max(acc_curr, res > magnitude)

            ind_curr = (acc_curr == 0).nonzero().squeeze()
            # acc[ind_to_fool[ind_curr]] = 0
            # adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

            acc[ind_curr] = 0
            adv[ind_curr] = adv_curr[ind_curr].clone()

    return adv,None,record_list

def record_fab_attack_single_run(x, y,model,magnitude,_type,use_rand_start=False, is_targeted=False,gpu_idx=None,n_iter=100,alpha_max=0.1,eta=1.05,beta=0.9):
    """
    :param x:             clean images
    :param y:             clean labels, if None we use the predicted labels
    :param is_targeted    True if we ise targeted version. Targeted class is assigned by `self.target_class`
    """
    device = 'cuda:{}'.format(gpu_idx)

    orig_dim = list(x.shape[1:])
    ndims = len(orig_dim)

    x = x.detach().clone().float().to(device)

    y_pred = fab_get_predicted_label(x,model) 
    if y is None:
        y = y_pred.detach().clone().long().to(device)
    else:
        y = y.detach().clone().long().to(device)
    pred = y_pred == y
    not_pred = y_pred!=y
    corr_classified = pred.float().sum()
    # if self.verbose:
    #     print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
    if pred.sum() == 0:
        return x
    pred = fab_check_shape(pred.nonzero().squeeze())

    startt = time.time()

    # runs the attack only on correctly classified points

    im2 = x[pred].detach().clone()
    la2 = y[pred].detach().clone() 


    if len(im2.shape) == ndims:
        im2 = im2.unsqueeze(0)
    bs = im2.shape[0] 
    u1 = torch.arange(bs) 
    adv = im2.clone() 
    adv_c = x.clone() 
    res2 = 1e10 * torch.ones([bs]).to(device) #?
    res_c = torch.zeros([x.shape[0]]).to(device)#[0,0,0,0,0,0,...,0]
    x1 = im2.clone()
    x0 = im2.clone().reshape([bs, -1])
    counter_restarts = 0

    while counter_restarts < 1:
        
        if use_rand_start:
            if _type == 'linf':
                t = 2 * torch.rand(x1.shape).to(device) - 1
                x1 = im2 + (torch.min(res2,
                                        magnitude * torch.ones(res2.shape)
                                        .to(device)
                                        ).reshape([-1, *[1]*ndims])
                            ) * t / (t.reshape([t.shape[0], -1]).abs()
                                        .max(dim=1, keepdim=True)[0]
                                        .reshape([-1, *[1]*ndims])) * .5
            elif _type == 'l2':
                t = torch.randn(x1.shape).to(device)
                x1 = im2 + (torch.min(res2,
                                        magnitude * torch.ones(res2.shape)
                                        .to(device)
                                        ).reshape([-1, *[1]*ndims])
                            ) * t / ((t ** 2)
                                        .view(t.shape[0], -1)
                                        .sum(dim=-1)
                                        .sqrt()
                                        .view(t.shape[0], *[1]*ndims)) * .5

            x1 = x1.clamp(0.0, 1.0)

        record_list=[]    

        counter_iter = 0
        while counter_iter < n_iter: 
            with torch.no_grad():
                df, dg = fab_get_diff_logits_grads_batch(x1, la2,model,gpu_idx)
                if _type == 'linf':
                    dist1 = df.abs() / (1e-12 +
                                        dg.abs()
                                        .reshape(dg.shape[0], dg.shape[1], -1)
                                        .sum(dim=-1))
                elif _type == 'l2':
                    dist1 = df.abs() / (1e-12 + (dg ** 2)
                                        .reshape(dg.shape[0], dg.shape[1], -1)
                                        .sum(dim=-1).sqrt())

                else:
                    raise ValueError('norm not supported')
                ind = dist1.min(dim=1)[1]
                dg2 = dg[u1, ind]
                b = (- df[u1, ind] + (dg2 * x1).reshape(x1.shape[0], -1)
                                        .sum(dim=-1))
                w = dg2.reshape([bs, -1])

                if _type == 'linf':
                    d3 = fab_projection_linf(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0))
                elif _type == 'l2':
                    d3 = projection_l2(
                        torch.cat((x1.reshape([bs, -1]), x0), 0),
                        torch.cat((w, w), 0),
                        torch.cat((b, b), 0))

                d1 = torch.reshape(d3[:bs], x1.shape)
                d2 = torch.reshape(d3[-bs:], x1.shape)
                if _type == 'linf':
                    a0 = d3.abs().max(dim=1, keepdim=True)[0]\
                        .view(-1, *[1]*ndims)
                elif _type == 'l2':
                    a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()\
                        .view(-1, *[1]*ndims)

                a0 = torch.max(a0, 1e-8 * torch.ones(
                    a0.shape).to(device))
                a1 = a0[:bs]
                a2 = a0[-bs:]
                alpha = torch.min(torch.max(a1 / (a1 + a2),
                                            torch.zeros(a1.shape)
                                            .to(device)),
                                    alpha_max * torch.ones(a1.shape)
                                    .to(device))
                x1 = ((x1 + eta * d1) * (1 - alpha) +
                        (im2 + d2 * eta) * alpha).clamp(0.0, 1.0)

                is_adv = fab_get_predicted_label(x1,model) != la2 


                if is_adv.sum() > 0:
                    ind_adv = is_adv.nonzero().squeeze()
                    ind_adv = fab_check_shape(ind_adv)
                    if _type == 'linf':
                        t = (x1[ind_adv] - im2[ind_adv]).reshape(
                            [ind_adv.shape[0], -1]).abs().max(dim=1)[0]
                    elif _type == 'l2':
                        t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                            .reshape(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                    
                    adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                        float().reshape([-1, *[1]*ndims]) + adv[ind_adv]\
                        * (t >= res2[ind_adv]).float().reshape(
                        [-1, *[1]*ndims]) 

                    res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                        + res2[ind_adv] * (t >= res2[ind_adv]).float()
                    x1[ind_adv] = im2[ind_adv] + (
                        x1[ind_adv] - im2[ind_adv]) * beta




                ind_succ = res2 < 1e10
                ind_succ = fab_check_shape(ind_succ.nonzero().squeeze())
                acc_curr = model(adv).max(1)[1] == la2
                if _type == 'linf':
                    res = (im2 - adv).abs().reshape(im2.shape[0], -1).max(1)[0]
                elif _type == 'l2':
                    res = ((im2 - adv) ** 2).reshape(im2.shape[0], -1).sum(dim=-1).sqrt()

                acc_curr = torch.max(acc_curr, res > magnitude).cpu().numpy()


                record = np.ones(len(y))
                record[pred[ind_succ.cpu().numpy()].cpu().numpy()] = acc_curr[ind_succ.cpu().numpy()]
                record[not_pred.cpu().numpy()]=0
                record_list.append(record)

                counter_iter += 1

        counter_restarts += 1

    ind_succ = res2 < 1e10



    res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
    ind_succ = fab_check_shape(ind_succ.nonzero().squeeze())
    adv_c[pred[ind_succ]] = adv[ind_succ].clone()




    return adv_c,record_list



def fab_get_diff_logits_grads_batch(imgs, la,model,gpu_idx):
    device = 'cuda:{}'.format(gpu_idx)
    im = imgs.clone().requires_grad_()
    with torch.enable_grad():
        y = model(im)

    g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(device)
    grad_mask = torch.zeros_like(y)
    for counter in range(y.shape[-1]):
        zero_gradients(im)
        grad_mask[:, counter] = 1.0
        y.backward(grad_mask, retain_graph=True)
        grad_mask[:, counter] = 0.0
        g2[counter] = im.grad.data

    g2 = torch.transpose(g2, 0, 1).detach()
    #y2 = self.predict(imgs).detach()
    y2 = y.detach()
    df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
    dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
    df[torch.arange(imgs.shape[0]), la] = 1e10

    return df, dg

def fab_get_predicted_label(x,model):
    with torch.no_grad():
        outputs = model(x)
    _, y = torch.max(outputs, dim=1)
    return y

def fab_check_shape(x):
    return x if len(x.shape) > 0 else x.unsqueeze(0)



def PGD_Attack_adaptive_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    # print(x.shape)

    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        if target is not None:
            loss_indiv = -F.cross_entropy(logits, target, reduce=False)
        else:
            loss_indiv = F.cross_entropy(logits, y, reduce=False)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            if target is not None:
                loss_indiv = -F.cross_entropy(logits, target, reduce=False)
            else:
                loss_indiv = F.cross_entropy(logits, y, reduce=False)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv[ind_non_suc] = x_best_adv
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def Record_PGD_Attack_adaptive_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    
    model.eval() 

    device = 'cuda:{}'.format(gpu_idx) 
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone() 
    pred = predict_from_logits(model(x)) 
    if torch.sum((pred==y)).item() == 0: 
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze() 

    ind_suc = (pred!=y).nonzero().squeeze()
    record_list = []

    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)


    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        if target is not None:
            loss_indiv = -F.cross_entropy(logits, target, reduce=False)
        else:
            loss_indiv = F.cross_entropy(logits, y, reduce=False)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            pred_after_attack = predict_from_logits(logits)
            record = np.ones(len(pred_after_attack))
            record = record * (pred_after_attack==y).cpu().numpy() 
            record_list.append(record)

            if target is not None:
                loss_indiv = -F.cross_entropy(logits, target, reduce=False)
            else:
                loss_indiv = F.cross_entropy(logits, y, reduce=False)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)

    adv[ind_non_suc] = x_best_adv[ind_non_suc]

    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    for item in record_list:
        item[ind_suc.cpu().numpy()]=0

    return adv, now_p, record_list


def DDNL2Attack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    batch_size = x.shape[0]
    data_dims = (1,) * (x.dim() - 1)
    norm = torch.full((batch_size,), 1, dtype=torch.float).to(device)
    worst_norm = torch.max(x - 0, 1 - x).flatten(1).norm(p=2, dim=1)

    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=0.01)

    best_l2 = worst_norm.clone()
    best_delta = torch.zeros_like(x)

    for i in range(max_iters):
        l2 = delta.data.flatten(1).norm(p=2, dim=1)
        logits = model(x + delta)
        pred_labels = logits.argmax(1)
        
        if target is not None:
            loss = F.cross_entropy(logits, target)
        else:
            loss = -F.cross_entropy(logits, y)

        is_adv = (pred_labels == target) if target is not None else (
            pred_labels != y)
        is_smaller = l2 < best_l2
        is_both = is_adv * is_smaller
        best_l2[is_both] = l2[is_both]
        best_delta[is_both] = delta.data[is_both]

        optimizer.zero_grad()
        loss.backward()

        # renorming gradient
        grad_norms = delta.grad.flatten(1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, *data_dims))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(
                delta.grad[grad_norms == 0])

        optimizer.step()
        scheduler.step()

        norm.mul_(1 - (2 * is_adv.float() - 1) * 0.05)

        delta.data.mul_((norm / delta.data.flatten(1).norm(
            p=2, dim=1)).view(-1, *data_dims))

        delta.data.add_(x)
        delta.data.mul_(255).round_().div_(255)
        delta.data.clamp_(0, 1).sub_(x)
        # print(best_l2)

    adv_imgs = x + best_delta

    dist = (adv_imgs - x)
    dist = dist.view(x.shape[0], -1)
    dist_norm = torch.norm(dist, dim=1, keepdim=True)
    mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
    dist = dist / dist_norm
    dist *= max_eps
    dist = dist.view(x.shape)
    adv_imgs = (x + dist) * mask.float() + adv_imgs * (1 - mask.float())

    if previous_p is not None:
        original_image = x - previous_p
        global_dist = adv_imgs - original_image
        global_dist = global_dist.view(x.shape[0], -1)
        dist_norm = torch.norm(global_dist, dim=1, keepdim=True)
        # print(dist_norm)
        mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
        global_dist = global_dist / dist_norm
        global_dist *= max_eps
        global_dist = global_dist.view(x.shape)
        adv_imgs = (original_image + global_dist) * mask.float() + adv_imgs * (1 - mask.float())
    
    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p


def RecordDDNL2Attack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()

    ind_suc = (pred!=y).nonzero().squeeze()
    record_list = []

    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    batch_size = x.shape[0]
    data_dims = (1,) * (x.dim() - 1)
    norm = torch.full((batch_size,), 1, dtype=torch.float).to(device)
    worst_norm = torch.max(x - 0, 1 - x).flatten(1).norm(p=2, dim=1)

    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=0.01)

    best_l2 = worst_norm.clone()
    best_delta = torch.zeros_like(x)

    for i in range(max_iters):
        l2 = delta.data.flatten(1).norm(p=2, dim=1)
        logits = model(x + delta)
        pred_labels = logits.argmax(1)
        
        if target is not None:
            loss = F.cross_entropy(logits, target)
        else:
            loss = -F.cross_entropy(logits, y)

        is_adv = (pred_labels == target) if target is not None else (
            pred_labels != y)
        is_smaller = l2 < best_l2
        is_both = is_adv * is_smaller
        best_l2[is_both] = l2[is_both]
        best_delta[is_both] = delta.data[is_both]

        optimizer.zero_grad()
        loss.backward()

        # renorming gradient
        grad_norms = delta.grad.flatten(1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, *data_dims))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(
                delta.grad[grad_norms == 0])

        optimizer.step()
        scheduler.step()

        norm.mul_(1 - (2 * is_adv.float() - 1) * 0.05)

        delta.data.mul_((norm / delta.data.flatten(1).norm(
            p=2, dim=1)).view(-1, *data_dims))

        delta.data.add_(x)
        delta.data.mul_(255).round_().div_(255)
        delta.data.clamp_(0, 1).sub_(x)
        # print(best_l2)
        adv_imgs = x + best_delta 

        dist = (adv_imgs - x)
        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
        dist = dist / dist_norm
        dist *= max_eps
        dist = dist.view(x.shape)
        adv_imgs = (x + dist) * mask.float() + adv_imgs * (1 - mask.float())   #?

        logits = model(adv_imgs)
        pred_after_attack = predict_from_logits(logits)
        record = np.ones(len(pred_after_attack))
        record = record * (pred_after_attack==y).cpu().numpy() 
        record_list.append(record)

    if previous_p is not None: #None
        original_image = x - previous_p
        global_dist = adv_imgs - original_image
        global_dist = global_dist.view(x.shape[0], -1)
        dist_norm = torch.norm(global_dist, dim=1, keepdim=True)
        # print(dist_norm)
        mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
        global_dist = global_dist / dist_norm
        global_dist *= max_eps
        global_dist = global_dist.view(x.shape)
        adv_imgs = (original_image + global_dist) * mask.float() + adv_imgs * (1 - mask.float())
    
    now_p = adv_imgs-x
    

    adv[ind_non_suc] = adv_imgs[ind_non_suc] 

    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    for item in record_list:
        item[ind_suc.cpu().numpy()]=0

    return adv, now_p ,record_list

def attacker_list():  
    l = [
         MultiTargetedAttack,
         RecordMultiTargetedAttack,

         CW_Attack_adaptive_stepsize,
         Record_CW_Attack_adaptive_stepsize,

         ApgdCeAttack,
         RecordApgdCeAttack,

         ApgdDlrAttack,
         RecordApgdDlrAttack,
         
         FabAttack,
         RecordFabAttack,

         PGD_Attack_adaptive_stepsize,
         Record_PGD_Attack_adaptive_stepsize,

         DDNL2Attack,
         RecordDDNL2Attack,
    ]
    return l


attacker_dict = {fn.__name__: fn for fn in attacker_list()}

def get_attacker(name):
    return attacker_dict[name]

def apply_attacker(img, name, y, model, magnitude, p, steps, max_eps, target=None, _type=None, gpu_idx=None):

    augment_fn = get_attacker(name)
    return augment_fn(x=img, y=y, model=model, magnitude=magnitude, previous_p=p, max_iters=steps,max_eps=max_eps, target=target, _type=_type, gpu_idx=gpu_idx)