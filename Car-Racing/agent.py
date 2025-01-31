import torch as T
from models import CartPoleDqn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn import MSELoss, HuberLoss


def get_discounted_table(n, v):
    tab = T.arange(n).unsqueeze(1) - T.arange(n)
    tab = v ** tab
    return tab.tril()


class EligibilityQAgent:
    def __init__(self, n_observations:int = 4, n_actions: int = 2, timesteps: int = 5, hidden_dim: int = 64, device = 'cpu'):
        self.tau_ = 0.005 # best 0.01
        self.lambda_ = 0.9
        self.gamma_ = 0.9
        self.local_agent = CartPoleDqn(
            n_observations, 
            n_actions, 
            hidden_dim, 
            # init_weight=1/(1-self.gamma_)-10
            ).to(device)
        self.target_agent = CartPoleDqn(n_observations, n_actions, hidden_dim).to(device)
        self.target_agent.load_state_dict(self.local_agent.state_dict())
        self.device = device
        self.optimizer = Adam(self.local_agent.parameters(), lr=0.0002, weight_decay=0.01) # best lr=0.0002, weight_decay=0.001
        
        # lambda1 = lambda i: (int(i>1000) * 0.1 * 0.999 ** (i - 1000) + int(i<=1000))
        # best self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=0.000001)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10000, T_mult=2, eta_min=0.000001)
        self.loss_fn = HuberLoss()
        self.discounted_table = get_discounted_table(timesteps, self.gamma_ * self.lambda_).to(device)
        self.upper_triangle = T.ones(timesteps, timesteps).triu().to(device)
        
    def update_target_weights(self):
        local_net_state_dict = self.local_agent.state_dict()
        target_net_state_dict = self.target_agent.state_dict()
        for key in local_net_state_dict:
            target_net_state_dict[key] = local_net_state_dict[key]*self.tau_ + target_net_state_dict[key]*(1-self.tau_)
        self.target_agent.load_state_dict(target_net_state_dict)
        
    def action(self, state, strategy: str = "softmax", temperature: float = 1):
        state = state.to(self.device)
        with T.no_grad():
            logits = self.local_agent(state)
        if strategy == "softmax":
            logits = logits / temperature
            m = Categorical(logits=logits)
            return m.sample()
        elif strategy == 'argmax':
            action = logits.argmax(-1)
            return action
        return -1
    
    def calc_one_step_td_error(self, state, next_state, action, reward, terminated):
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device).unsqueeze(-1)
        reward = reward.to(self.device)
        terminated = terminated.to(self.device).to(T.float)
        with T.no_grad():
            next_state_q_values = self.target_agent(next_state)
            state_q_values = self.local_agent(state)
        action_q_value = state_q_values.gather(dim=-1, index=action)
        action_q_value = action_q_value.squeeze(-1)
        next_state_value = next_state_q_values.max(-1).values
        td_error = reward + self.gamma_ * next_state_value * (1 - terminated) - action_q_value
        sign_td_error = (td_error > 1).to(T.float)
        
        final_td_error = sign_td_error * td_error.pow(2) + (1 - sign_td_error) * T.abs(td_error)
        return final_td_error.cpu().numpy()
    
    def train(self, sample):
        self.optimizer.zero_grad()
        states, actions, rewards, terminateds = [x.to(self.device) for x in sample]
        next_states = states[:, 1:].detach()
        states = states[:, :-1]
        actions = actions.unsqueeze(-1)
        
        with T.no_grad():
            next_states_logits = self.target_agent(next_states)
        next_states_values = next_states_logits.max(-1).values
        
        
        logits = self.local_agent(states)
        action_logits = logits.gather(dim=-1, index=actions).squeeze(-1)
        
        terminateds = terminateds.to(T.float)
        following_terminateds = T.matmul(terminateds, self.upper_triangle)
        
        targets = - 10 / (1 - self.gamma_ * 0) * terminateds + rewards + self.gamma_ * next_states_values * (1 - terminateds)
        td_errors = targets * (1 - self.gamma_ * 0) - action_logits
        
        td_errors = td_errors * (1 - following_terminateds + terminateds)
        discounted_td_errors = T.matmul(td_errors, self.discounted_table)
        sign_td_error = (discounted_td_errors > 1).to(T.float)
        
        loss = sign_td_error * discounted_td_errors.pow(2) + (1 - discounted_td_errors) * T.abs(discounted_td_errors)
        
        loss = loss.sum(-1)
        loss = loss.mean()
        loss.backward()
        T.nn.utils.clip_grad_value_(self.local_agent.parameters(), 10)
        self.optimizer.step()
        self.step_scheduler()
        
        self.update_target_weights()
        return loss.item() 
    
    def step_scheduler(self):
        self.scheduler.step()
