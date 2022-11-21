import pickle
import numpy as np
import copy
def parallel_get_attacker_accuracy_from_record_list(new_attack):
    
    attack_name = new_attack['attacker']
    attack_eps = new_attack['magnitude']
    attack_steps = new_attack['step']
    if attack_name=='RecordMultiTargetedAttack_L2':
        with open('./record_list/Record_list_RecordMultiTargetedAttack_L2.pkl', 'rb') as f:
            record_list = pickle.load(f)
    elif attack_name=='Record_CW_Attack_L2_adaptive_stepsize':
        with open('./record_list/Record_list_Record_CWAttack_adaptive_stepsize_L2.pkl', 'rb') as f:
            record_list = pickle.load(f)
    elif attack_name=='RecordApgdCeAttack':
        with open('./record_list/Record_list_RecordApgdCeAttack_L2.pkl', 'rb') as f:
            record_list = pickle.load(f)
    elif attack_name=='RecordApgdDlrAttack':
        with open('./record_list/Record_list_RecordApgdDlrAttack_L2.pkl', 'rb') as f:
            record_list = pickle.load(f)
    elif attack_name=='RecordFabAttack':
        with open('./record_list/Record_list_RecordFabAttack_L2.pkl', 'rb') as f:
            record_list = pickle.load(f)
    elif attack_name=='Record_PGD_Attack_adaptive_stepsize':
        with open('./record_list/Record_list_Record_PGD_Attack_adaptive_stepsize_L2.pkl', 'rb') as f:
            record_list = pickle.load(f)
    elif attack_name=='RecordDDNL2Attack':
        with open('./record_list/Record_list_RecordDDNL2Attack_L2.pkl', 'rb') as f:
            record_list = pickle.load(f)       
    acc_total = record_list[attack_steps]
    return acc_total
def parallel_append_next_attack_from_record_list(last_acc_total,t_max):
    max_result = 0
    original_accuracy = np.array(last_acc_total).sum()
    best_attacker = None
    best_acc_total = None
    best_t = None
    acc_total = copy.deepcopy(last_acc_total) 
    for attack_idx in range(len(candidate_pool)):
        new_attack = candidate_pool[attack_idx]
        attack_name = new_attack['attacker']
        if attack_name=='RecordMultiTargetedAttack_L2':
            tmp_t_max = min(504,t_max)
            tmp_t_start = 63-1
            tmp_t_add = 63
        elif attack_name=='Record_CW_Attack_L2_adaptive_stepsize':
            tmp_t_max = min(1000,t_max)
            tmp_t_start = 125-1
            tmp_t_add = 125
        elif attack_name=='RecordApgdCeAttack':
            tmp_t_max = min(256,t_max)
            tmp_t_start = 32-1
            tmp_t_add = 32
        elif attack_name=='RecordApgdDlrAttack':
            tmp_t_max = min(256,t_max)
            tmp_t_start = 32-1
            tmp_t_add = 32
        elif attack_name=='RecordFabAttack':
            tmp_t_max = min(504,t_max)
            tmp_t_start = 63-1
            tmp_t_add = 63
        elif attack_name=='Record_PGD_Attack_adaptive_stepsize':
            tmp_t_max = min(1000,t_max)
            tmp_t_start = 125-1
            tmp_t_add = 125
        elif attack_name=='RecordDDNL2Attack':
            tmp_t_max = min(1000,t_max)
            tmp_t_start = 125-1
            tmp_t_add = 125
        

        for t in range(tmp_t_start,tmp_t_max,tmp_t_add):
            new_attack['step']= t 
            tmp_acc_total = parallel_get_attacker_accuracy_from_record_list(new_attack)
            last_acc_total = list(map(int,last_acc_total))
            tmp_acc_total = list(map(int,tmp_acc_total))
            tmp_new_policy_acc_total = np.bitwise_and(last_acc_total,tmp_acc_total)
            cur_result = (original_accuracy-tmp_new_policy_acc_total.sum())/t
            if cur_result>max_result:
                best_t = copy.deepcopy(t)
                best_acc_total = copy.deepcopy(tmp_new_policy_acc_total)
                best_attacker = copy.deepcopy(new_attack)
                max_result = cur_result
                #print("find best:",best_attacker)
    return [best_attacker,best_acc_total,best_t]

def parallel_greedy_algorithm_from_record_list():
    policy = []
    policy_acc_total = np.ones(5000)
    t_max = 1000 
    while t_max >= 0:
        #print("remaining t:{}",t_max)
        #print("trying to get next attack...")
        [next_attack, policy_acc_total, t] = parallel_append_next_attack_from_record_list(policy_acc_total, t_max)
        if next_attack is None:
            return policy
        policy.append(next_attack)
        t_max = t_max - t
        #print("final accuacy:",policy_acc_total.sum())
    return policy



RecordMultiTargetedAttack_L2 = {'attacker': 'RecordMultiTargetedAttack_L2', 'magnitude': 0.5, 'step': 50}
Record_CWAttack_adaptive_stepsize_L2 = {'attacker': 'Record_CW_Attack_L2_adaptive_stepsize', 'magnitude': 0.5, 'step': 50}
Record_PGD_Attack_adaptive_stepsize_L2 = {'attacker': 'Record_PGD_Attack_adaptive_stepsize', 'magnitude': 0.5, 'step': 50}
RecordDDNL2Attack_L2 = {'attacker': 'RecordDDNL2Attack', 'magnitude': None, 'step': 50}
RecordApgdCeAttack_L2 = {'attacker': 'RecordApgdCeAttack', 'magnitude': 0.5, 'step': 50}
RecordApgdDlrAttack_L2 ={'attacker': 'RecordApgdDlrAttack', 'magnitude': 0.5, 'step': 50}
RecordFabAttack_L2 = {'attacker': 'RecordFabAttack', 'magnitude': 0.5, 'step': 50}
candidate_pool = [ RecordMultiTargetedAttack_L2, Record_CWAttack_adaptive_stepsize_L2, RecordApgdCeAttack_L2, RecordApgdDlrAttack_L2, RecordFabAttack_L2,Record_PGD_Attack_adaptive_stepsize_L2,RecordDDNL2Attack_L2]
result = parallel_greedy_algorithm_from_record_list()
print(result)

