import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from .dmaq_qatten_weight import Qatten_Weight
from .dmaq_si_weight import DMAQ_SI_Weight, DMAQ_A_SI_Weight
from sklearn.covariance import shrunk_covariance

class DMAQ_QattenMixer(nn.Module):
    def __init__(self, args):
        super(DMAQ_QattenMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.attention_weight = Qatten_Weight(args)
        self.si_weight = DMAQ_SI_Weight(args,0)
        #self.var = DMAQ_SI_Weight(args,0)
        
        #self.cov = DMAQ_A_SI_Weight(args)
        #self.cov = [DMAQ_SI_Weight(args,1) for i in range(self.n_agents)]
    
    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final_temp = self.si_weight(states, actions)
        adv_w_final_mean = adv_w_final_temp[:,:self.n_agents].clone()
        var = adv_w_final_temp[:,self.n_agents:].clone()

        std = th.exp(0.5 * var)
        eps = th.randn_like(std)
        samples = adv_w_final_mean + eps * std
        
        adv_w_final = samples.view(-1, self.n_agents)
        
        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        var = th.sum(var,dim=1)
        return adv_tot, var

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot,0
        else:
            adv_tot,var = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot,var

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents)  + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y,var = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies,0,var
