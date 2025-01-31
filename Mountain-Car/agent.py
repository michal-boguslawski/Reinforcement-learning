import torch as T
from models import MountainCarModel
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn import MSELoss, HuberLoss


def get_discounted_table(n, v):
    tab = T.arange(n).unsqueeze(1) - T.arange(n)
    tab = v ** tab
    return tab.tril()

def entropy(std):
    return 0.5 * T.log(2 * T.pi * T.e * std ** 2)

       
class ACAgent:
    def __init__(self, input_dim: int = 2, output_dim: int = 1, timesteps: int = 5, hidden_dim: int = 64, device = 'cpu'):
        self.tau_ = 0.005 # best 0.01
        self.lambda_ = 0.99
        self.gamma_ = 0.9
        self.agent = MountainCarModel(
            input_dim, 
            output_dim, 
            hidden_dim
            ).to(device)
        self.target_agent.load_state_dict(self.local_agent.state_dict())
        self.device = device
        self.optimizer = Adam(self.local_agent.parameters(), lr=0.0002, weight_decay=0.) # best lr=0.0002, weight_decay=0.001
        
        # lambda1 = lambda i: (int(i>1000) * 0.1 * 0.999 ** (i - 1000) + int(i<=1000))
        # best self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=0.000001)
        self.scheduler = None # CosineAnnealingWarmRestarts(self.optimizer, T_0=10000, T_mult=2, eta_min=0.000001)
        self.loss_fn = HuberLoss()
        self.discounted_table = get_discounted_table(timesteps, self.gamma_ * self.lambda_).to(device)
        self.upper_triangle = T.ones(timesteps, timesteps).triu().to(device)
        
        
    def action(self, state, strategy: str = "softmax", temperature: float = 1):
        state = state.to(self.device)
        with T.no_grad():
            action_mean, action_std, _ = self.local_agent(state)
        rnd = T.randn_like(action_mean)
        action = action_mean + action_std * rnd
        return action.clip(-1, 1)
    
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
        actions = actions.unsqueeze(-1)
        
        actions_mean, actions_std, states_value = self.agent(states)
        next_states_value = states_value[:, 1:].clone().detach()
        states_value = states_value[:, :-1]
        
        terminateds = terminateds.to(T.float)
        following_terminateds = T.matmul(terminateds, self.upper_triangle)
        
        td_errors = rewards + self.gamma_ * next_states_value * (1 - terminateds) - states_value
        discounted_td_errors = T.matmul(td_errors, self.discounted_table) * following_terminateds
        
        policy_loss = - (discounted_td_errors * T.log(actions_mean / 2 + 0.5 + 1e-8)).sum(-1)
        
        sign_td_error = (discounted_td_errors > 1).to(T.float)
        
        critic_loss = sign_td_error * discounted_td_errors.pow(2) + (1 - discounted_td_errors) * T.abs(discounted_td_errors)
        critic_loss = critic_loss.sum(-1)
        
        entropy = entropy(actions_std).sum(-1)
        
        loss = policy_loss + 0.5 * critic_loss + 0.0001 * entropy
        loss = loss.mean()
        loss.backward()
        T.nn.utils.clip_grad_value_(self.agent.parameters(), 10)
        self.optimizer.step()
        if self.scheduler:
            self.step_scheduler()
        return loss.item() 
    
    def step_scheduler(self):
        self.scheduler.step()
