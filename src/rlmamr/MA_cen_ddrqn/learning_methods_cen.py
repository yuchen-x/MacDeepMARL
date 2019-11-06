import numpy as np
import IPython
import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from IPython.core.debugger import set_trace
from .utils.utils import _prune_o, _prune_o_n, _prune_filtered_o, _prune_filtered_o_n

def QLearn_squ_cen_0(cen_controller, batch, hysteretic, discount, trace_len, sub_trace_len, batch_size, 
                     sort_traj=False, huber_loss=False, grad_clip=False, grad_clip_value=None,
                     grad_clip_norm=False, grad_clip_max_norm=None, rnn=True, device='cpu', **kwargs):

    # calculate loss for controller
    policy_net = cen_controller.policy_net
    target_net = cen_controller.target_net

    # seperate elements in the batch
    state_b, action_b, id_reward_b, j_reward_b, state_next_b, terminate_b, id_valid_b, j_valid_b = zip(*batch)

    assert len(state_b) == trace_len * batch_size, "batch's data problem ..."
    assert len(state_next_b) == trace_len * batch_size, "batch's data problem ..."

    s_b = torch.cat(state_b).view(batch_size, trace_len, -1).to(device)             #dim: (batch_size, trace_len, policy.net.input_dim)
    a_b = torch.cat(action_b).view(batch_size, trace_len, 1).to(device)             #dim: (batch_size, trace_len, 1)
    j_r_b = torch.cat(j_reward_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
    s_next_b = torch.cat(state_next_b).view(batch_size, trace_len, -1).to(device)   #dim: (batch_size, trace_len, policy.net.input_dim)
    t_b = torch.cat(terminate_b).view(batch_size, trace_len, 1).to(device)          #dim: (batch_size, trace_len, 1)
    j_v_b = torch.cat(j_valid_b).view(batch_size, trace_len).to(device)             #dim: (batch_size, trace_len)

    if sort_traj:
        
        sorted_index = j_v_b.sum(1).sort(0, descending=True)[1].view(-1)    # sort mask(v_b) depeding on the length of each trajectory
        sorted_s_b = s_b[sorted_index]                                      # sort s_b according to the sorted mask v_b
        sorted_a_b = a_b[sorted_index]                                      # sort a_b according to the sorted mask v_b
        sorted_j_r_b = j_r_b[sorted_index]                                  # sort j_r_b according to the sorted mask v_b
        sorted_s_next_b = s_next_b[sorted_index]                            # sort s_next_b according to the sorted mask v_b
        sorted_t_b = t_b[sorted_index]                                      # sort t_b according to the sorted mask v_b
        sorted_j_v_b = j_v_b[sorted_index]                                  # sort j_v_b according to the sorted mask v_b

        selected_traj_mask = sorted_j_v_b.sum(1) >= sub_trace_len
        if torch.sum(selected_traj_mask).item() == 0:
            return
        selected_lengths = sorted_j_v_b.sum(1)[selected_traj_mask]
        
        s_b = torch.split_with_sizes(sorted_s_b[selected_traj_mask][sorted_j_v_b[selected_traj_mask]], list(selected_lengths))
        s_next_b = torch.split_with_sizes(sorted_s_next_b[selected_traj_mask][sorted_j_v_b[selected_traj_mask]], list(selected_lengths))
        a_b = sorted_a_b[selected_traj_mask][sorted_j_v_b[selected_traj_mask]]
        j_r_b = sorted_j_r_b[selected_traj_mask][sorted_j_v_b[selected_traj_mask]]
        t_b = sorted_t_b[selected_traj_mask][sorted_j_v_b[selected_traj_mask]]

    else:

        selected_traj_mask = j_v_b.sum(1) >= sub_trace_len
        if torch.sum(selected_traj_mask).item() == 0:
            return
        selected_lengths = j_v_b.sum(1)[selected_traj_mask]

        s_b = torch.split_with_sizes(s_b[selected_traj_mask][j_v_b[selected_traj_mask]], list(selected_lengths))
        s_next_b = torch.split_with_sizes(s_next_b[selected_traj_mask][j_v_b[selected_traj_mask]], list(selected_lengths))
        a_b = a_b[selected_traj_mask][j_v_b[selected_traj_mask]]
        j_r_b = j_r_b[selected_traj_mask][j_v_b[selected_traj_mask]]
        t_b = t_b[selected_traj_mask][j_v_b[selected_traj_mask]]

    if rnn:
        Q_s, _ = policy_net(s_b)
        Q_s_next, _ = policy_net(s_next_b)
    else:
        Q_s  = policy_net(s_b)
        Q_s_next  = policy_net(s_next_b)

    assert Q_s.size(0) == a_b.size(0), "number of Qs doesn't match with number of actions"

    Q = Q_s.gather(1, a_b)

    # apply double Q learning
    a_next_b = Q_s_next.max(1)[1].view(-1, 1)
    if rnn:
        target_Q_s_next = target_net(s_next_b)[0].detach()
    else:
        target_Q_s_next = target_net(s_next_b).detach()
    target_Q = target_Q_s_next.gather(1, a_next_b)
    target_Q = j_r_b + discount * target_Q * (-t_b + 1)

    if huber_loss:
        cen_controller.loss = F.smooth_l1_loss(Q, targetQ)
    else:
        td_err = (target_Q - Q) 
        td_err = torch.max(hysteretic*td_err, td_err)
        cen_controller.loss = torch.mean(td_err*td_err)
    
    # optimize params for each agent
    if cen_controller.loss is not None:
        cen_controller.optimizer.zero_grad()
        cen_controller.loss.backward()

        if grad_clip:
            assert grad_clip_value is not None, "no grad_clip_value is given"
            for param in cen_controller.policy_net.parameters():
                param.grad.data.clamp_(-grad_clip_value, grad_clip_value)
        elif grad_clip_norm:
            assert grad_clip_max_norm is not None, "no grad_clip_max_norm is given"
            clip_grad_norm_(cen_controller.policy_net.parameters(), grad_clip_max_norm)
        cen_controller.optimizer.step()
        cen_controller.loss = None
