from gymnasium.vector import VectorEnv
import logging
import numpy as np
import torch as T
from tqdm import tqdm
from typing import Any, Literal

from agent.base import BasePolicy
from agent.factories import get_policy
from envs.factories import make_vec
from envs.utils import get_env_vec_details
from evaluate.evaluate import Evaluator
from models.models import ActionSpaceType, EnvDetails
from network.model import RLModel
from .utils import get_device, prepare_action_for_env
# from .evaluate import record_episode


np.set_printoptions(linewidth=1000)
logger = logging.getLogger(__name__)


class Worker:
    env: VectorEnv
    action_space_type: ActionSpaceType
    env_details: EnvDetails
    agent: BasePolicy
    i: int
    losses_list: list[list[float] | float]
    train_step: int
    batch_size: int
    minibatch_size: int
    epsilon: float

    def __init__(
        self,
        experiment_name: str,
        env_config: dict[str, Any],
        policy_config: dict[str, Any],
        network_config: dict[str, Any] | None = None,
        device: T.device | Literal["auto", "cpu", "cuda"] = T.device("cpu"),
        record_step: int = 100_000,
        verbose: int = 0,
        temperature_config: dict[str, Any] | None = None,
        *args,
        **kwargs
    ):
        self.experiment_name = experiment_name
        self.device = get_device(device)
        logger.info(f"Current device {self.device}")
        self.record_step = record_step
        self.verbose = verbose
        self.policy_config = policy_config

        self._setup_env(env_config)
        self._setup_network(network_config or {}, self.device)
        self._setup_policy(policy_config, self.device)
        self._setup_temperature(temperature_config or {})

    def _setup_temperature(self, temperature_config: dict[str, Any]) -> None:
        self.temperature_start = temperature_config.get("temperature_start", 1)
        self.temperature_end = temperature_config.get("temperature_end", 1)
        self.temperature_steps = temperature_config.get("temperature_steps", 1)
        self.temperature = self.temperature_start

    def _setup_env(self, env_config: dict[str, Any]) -> None:
        self.env_config = env_config
        self.env = make_vec(verbose=self.verbose, normalize_gamma=self.policy_config.get("gamma", 0.99), **env_config)
        self.env_details = get_env_vec_details(self.env)
        self.action_space_type = self.env_details.action_space_type

    def _setup_network(self, network_config: dict[str, Any], device: T.device = T.device("cpu")) -> None:
        network_kwargs = network_config.get("kwargs", {})
        self.network = RLModel(
            input_shape=self.env_details.state_dim,
            num_actions=self.env_details.action_dim,
            low=self.env_details.action_low,
            high=self.env_details.action_high,
            device=device,
            **network_kwargs
        )

    def _setup_policy(self, policy_config: dict[str, Any], device: T.device = T.device("cpu")) -> None:
        policy_type = policy_config.get("type", "a2c")
        policy_kwargs = policy_config.get("kwargs", {})

        self.agent = get_policy(
            policy_type=policy_type,
            network=self.network,
            action_space_type=self.action_space_type,
            device=device,
            policy_kwargs=policy_kwargs
        )

    def _reset_training_vars(
        self,
        batch_size: int,
        minibatch_size: int,
        timesteps: int | None = None,
        train_step: int | None = None,
    ):
        self.train_step = train_step or batch_size
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.timesteps = timesteps
        self.core_state = None
        
        self.state, _ = self.env.reset()

    def _log_step(self, done: T.Tensor, terminated: T.Tensor, truncated: T.Tensor):
        if done.any():
            final_rewards = T.tensor(self.env.env.call("episode_returns"))
            final_lenghts = T.tensor(self.env.env.call("episode_lengths"))
            log = {
                "episode/n": done.cpu().sum().item(),
                "episode/terminated": terminated.cpu().sum().item(),
                "episode/truncated": truncated.cpu().sum().item(),
                "episode/mean_reward": final_rewards[done.cpu()].cpu().mean().item(),
                "episode/mean_length": final_lenghts[done.cpu()].float().cpu().mean().item(),
            }
            logger.debug(log)

    def _set_temperature(self, num_steps: int):
        self.temperature = self.temperature_start - (self.temperature_start - self.temperature_end) * num_steps / self.temperature_steps
        logger.debug({"hyperparameters/temperature": self.temperature})

    def _step(self):
        try:
            state = self.state.to(self.device)
            action_output = self.agent.action(state=state, core_state=self.core_state, temperature=self.temperature)
            action = action_output.action

            env_action = prepare_action_for_env(action, self.action_space_type)

            next_state, reward, terminated, truncated, info = self.env.step(env_action)
        except Exception as e:
            logger.error(f"Error during episode step: {e}")
            raise e
        
        done = T.logical_or(terminated, truncated)
        reward = T.as_tensor(reward, device=self.device)
        done = T.as_tensor(done, dtype=T.bool, device=self.device)
        action = T.as_tensor(action, device=self.device)
        core_state = (
            T.zeros_like(action_output.core_state) 
            if action_output.core_state is not None and self.core_state is None
            else self.core_state
        )
        
        record = {
            "state": state,
            "logits": action_output.logits,
            "action": action,
            "reward": reward,
            "done": done,
            "value": action_output.value,
            "log_probs": action_output.log_probs,
            "core_state": core_state,
            # "dist": action_output.dist,
        }
        
        self.agent.update_buffer(record)
        
        self.state = next_state
        self.core_state = action_output.core_state

        if self.verbose == 1:
            self._log_step(done, terminated, truncated)

        return done

    def _train_agent(self):
        self.agent.train(
            batch_size=self.batch_size,
            minibatch_size=self.minibatch_size,
            timesteps=self.timesteps,
            temperature=self.temperature,
        )

    def train(
        self,
        num_steps: int,
        batch_size: int,
        timesteps: int | None = None,
        minibatch_size: int = 64,
        train_step: int | None = None,
        *args,
        **kwargs
    ):
        logger.info(f"{20 * '='} Start training {20 * '='}")
        self._reset_training_vars(
            train_step=train_step,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            timesteps=timesteps,
        )
        
        tq_iter = tqdm(range(int(num_steps)), desc=f"Training {self.experiment_name}", unit="steps")
        
        for num_step in tq_iter:
            done = self._step()

            if (num_step % self.train_step == 0 and num_step > 0):
                self._train_agent()
                self._set_temperature(num_step)

            no_of_finished_envs = done.sum().cpu()
            if no_of_finished_envs:
                if self.core_state is not None:
                    self.core_state *= T.logical_not(done).unsqueeze(-1).expand_as(self.core_state)

                # tq_iter.set_postfix_str(f"temp mean rewards {temp_reward_mean:.2f}")

            if num_step % self.record_step == 0:
                self._print_on_record_step(num_step)

        self._print_on_record_step(num_steps)
        self._evaluate_policy()
        
        self.env.close()
        logger.info(f"{20 * '='} End training {20 * '='}")

    def _print_on_record_step(self, num_step: int):
        folder_path = f"logs/{self.experiment_name}"
        self.agent.save_weights(folder_path=folder_path)

        evaluator = Evaluator(
            **{
                **self.env_config,
                "num_envs": 1,
                "video_folder": f"logs/{self.experiment_name}/videos/step_{num_step}",
            }
        )
        
        self.agent.eval_mode()

        with evaluator:
            evaluator.evaluate(self.agent, min_episodes=1, action_space_type=self.action_space_type, temperature=0.0001)

        self.agent.train_mode()

    def _evaluate_policy(self):
        evaluator = Evaluator(record=False, **self.env_config)
        self.agent.eval_mode()

        with evaluator:
            evaluator.evaluate(self.agent, min_episodes=1000, action_space_type=self.action_space_type, temperature=0.0001)

        self.agent.train_mode()
