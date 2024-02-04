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
        self.var = DMAQ_SI_Weight(args,0)
        
        #self.cov = DMAQ_A_SI_Weight(args)
        self.cov = [DMAQ_SI_Weight(args,1) for i in range(self.n_agents)]
    
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

        adv_w_final = self.si_weight(states, actions)
        
        #cov = self.cov(states,actions)
        cov = [self.cov[i](states,actions)for i in range(self.n_agents)]
        cov = th.stack(cov,dim=1)

        # Clone the last dimension separately
        #decay = adv_w_final[:, -1].clone()
        mean = adv_w_final[:, :self.n_agents].clone()
        #var = adv_w_final[:,self.n_agents:].clone()
        var = self.var(states, actions)
        cov = cov.clone()
        cov_numpy= cov.detach().numpy().reshape(var.shape[0],self.n_agents, self.n_agents-1)

        var = var.clone()
        var_numpy= var.detach().numpy().reshape(var.shape[0],self.n_agents)
        
        #alpha = self.alpha(states, actions)
        #alpha_clone = alpha.clone()
        #alpha_numpy = alpha_clone.detach().numpy().reshape(alpha.shape[0],self.n_agents)

        #beta = self.beta(states, actions)
        #beta_clone = beta.clone()
        #beta_numpy = beta_clone.detach().numpy().reshape(alpha.shape[0],self.n_agents)
        
        from torch.distributions.multivariate_normal import MultivariateNormal
        cov_temp = cov.view(cov.shape[0],self.n_agents, self.n_agents-1)
        samples = th.zeros((mean.shape[0],self.n_agents))
        cov_new = th.zeros((cov.shape[0],self.n_agents,self.n_agents))
        for j in range(self.n_agents):
            cov_new[:,j,:j] = cov_temp[:,j,:j]
            cov_new[:,j,j] = var[:,j]
            cov_new[:,j,j+1:] = cov_temp[:,j,j:]

        '''
        samples_total = th.zeros(mean.shape[0],self.n_agents)
        for i in range(mean.shape[0]):
            m_total = MultivariateNormal(mean[i,:],scale_tril=th.tril(cov_new[i,:,:]))
            samples_total[i,:] = th.relu(m_total.sample((1,)))
        '''
        
        L = th.linalg.cholesky(cov_new)
        epsilon = th.randn_like(mean)
        z = mean + th.matmul(L, epsilon.unsqueeze(-1)).squeeze(-1)
        samples = th.tensor(z, dtype=th.float32, requires_grad=True)
        samples = th.nn.functional.relu(samples)
        
        '''
        for k in range(self.n_agents):
            # delete row
            cov_shrink  = th.cat((cov_new[:, :k, :], cov_new[:, k + 1:, :]), dim=1)
            # delete column
            cov_shrink = th.cat((cov_shrink[:, :, :k], cov_shrink[:, :,k+1 :]), dim=2)
            #delete mean
            mean_shrink = th.cat((mean[:,:k], mean[:,k+1 :]), dim=1)
            for i in range(mean.shape[0]):
                m_i = MultivariateNormal(mean_shrink[i,:],scale_tril=th.tril(cov_shrink[i,:]))
                sample_i = th.relu(m_i.sample((1,)))

                s1 = th.sum(samples_total,dim=2)
                s2 = th.sum(sample_i,dim=2)
            
                s_t = s1
                samples[i,k] = th.sigmoid(s_t)
        '''
        '''
        mean_numpy = mean.detach().numpy()
        samples = np.zeros((mean_numpy.shape[0],mean_numpy.shape[1]))
        for i in range(cov.shape[0]):
            # Extract the diagonal elements
            cov_new = np.zeros((self.n_agents,self.n_agents))
            for j in range(self.n_agents):
                cov_new[j][:j] = cov_numpy[i,j,:j]
                cov_new[j][j] = var_numpy[i][j]
                #cov_new[j][j] = np.tan(var_numpy[i][j])-var_numpy[i][j]
                cov_new[j][j+1:] = cov_numpy[i,j,j:]
                
            samples_total = np.random.multivariate_normal(mean_numpy[i],cov_new,1)
            #samples = np.sum(samples_total,axis=0)
            samples_total = np.maximum(samples_total,0)
            samples[i,:] = samples_total
            #samples = np.mean(samples,axis=0)
            #samples = np.sum(samples_total,axis=0)

            #for k in range(self.n_agents):
            #    # delete row
            #    cov_temp = np.delete(cov_new,k,0)
                
            #    # delete columm
            #    cov_temp = np.delete(cov_temp,k,1)
            #    mean_temp = np.delete(mean_numpy[i],k)
            #    samples_i = np.random.multivariate_normal(mean_temp,cov_temp,1)
            #    samples[i][k] = np.sum(samples_i)
                
            #    # delete columm
                

        #for i in range(mean_numpy.shape[0]):
        #    for j in range(mean_numpy.shape[1]):
        #        #samples[i][j] = samples[i][j] +alpha_numpy[i][j]*mean_numpy[i][j]
        #        samples[i][j] = samples[i][j] +mean_numpy[i][j]

        #dist = th.distributions.Normal(mean,std)
        #dist = th.distributions.dirichlet.Dirichlet(adv_w_final)
        #samples = dist.sample()
        #samples.requires_grad=True
        samples = th.tensor(samples, dtype=th.float32, requires_grad=True)
        samples = th.nn.functional.relu(samples)
        '''
        
        #adv_w_final = adv_w_final.view(-1, self.n_agents)
        adv_w_final = samples.view(-1, self.n_agents)
        #adv_w_final = adv_w_final[:,:-1].view(-1, self.n_agents)
        
        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            decay = 0
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

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

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies,0
