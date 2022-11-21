# Repository of the Paper "Reliable Robustness Evaluation via Automatically Constructed Attack Ensembles"
This repository contains the source code for paper: Reliable Robustness Evaluation via Automatically Constructed Attack Ensembles

## Pre-requisites
* torch = 1.7.1

* torchvision = 0.8.2

* advertorch = = 0.2.3

* tqdm = 4.56.2

* pillow = 5.4.1

* imagenet_c = 0.0.3

## Usage
### Retrive the Candidate Attacks' Performance Data
 `python get_record_list.py --batch_size 64 --dataset cifar10 --net_type madry_adv_resnet50_l2 --norm l2 --max_epsilon 0.5 --l2_attacker RecordDDNL2Attack_L2`

### Run AutoAE to build Linf Attack Ensemble (AE)
Collect the [training model](https://www.dropbox.com/s/c9qlt1lbdnu9tlo/cifar_linf_8.pt?dl=0) and place it to `/checkpoints`, and then run
`python get_linf_policy`

### Run AutoAE to build L2 Attack Ensemble (AE)
Collect the [training model](https://www.dropbox.com/s/1zazwjfzee7c8i4/cifar_l2_0_5.pt?dl=0) and place it to `/checkpoints`, and then run
`python get_l2_policy`

### Attack a denfense model with AEs
* Download a defense model
* Change get_robust_accuracy_by_AutoAE.py to load your defense model, and run:

    `python get_robust_accuracy_by_AutoAE.py --batch_size 8 --dataset cifar10 --net_type madry_adv_resnet50 --norm linf `

## Document Directory Structure
```
└─src
    │  attack_ops.py
    │  attack_utils.py
    │  fab_projections.py
    │  get_l2_policy.py
    │  get_linf_policy.py
    │  get_record_list.py
    │  get_robust_accuracy_by_AutoAE.py
    │  README.md
    │  tv_utils.py
    │
    ├─checkpoints
    │      cifar_l2_0_5.pt
    │      cifar_linf_8.pt
    │
    ├─cifar_models
    │      resnet.py
    │
    └─record_list
            Record_list_RecordApgdCeAttack_L2.pkl
            Record_list_RecordApgdCeAttack_Linf.pkl
            Record_list_RecordApgdDlrAttack_L2.pkl
            Record_list_RecordApgdDlrAttack_Linf.pkl
            Record_list_RecordDDNL2Attack_L2.pkl
            Record_list_RecordFabAttack_L2.pkl
            Record_list_RecordFabAttack_Linf.pkl
            Record_list_RecordMultiTargetedAttack_L2.pkl
            Record_list_RecordMultiTargetedAttack_Linf.pkl
            Record_list_Record_CWAttack_adaptive_stepsize_L2.pkl
            Record_list_Record_CWAttack_adaptive_stepsize_Linf.pkl
            Record_list_Record_PGD_Attack_adaptive_stepsize_L2.pkl
```