# Reinforcement learning using Neural Networks

## Cart Pole
### Problem Description
Model used: DQN

## Example of an episode
[![Watch the video](https://dagshub.com/boguslawski.m.j/Reinforcement-learning/src/main/Cart-Pole/CartPoleGame.gif)](https://dagshub.com/boguslawski.m.j/Reinforcement-learning/src/main/Cart-Pole/Cart-Pole-example.mp4)


## Mountain Car
### Problem Description
Model used: A2C with multiple environments and intrinsic reward.
Mean result achieved over 100 tries: 96.41

Two runs:
First with reward reshape presented below and over 1mln steps
def reshape_reward(reward, action, velocity, next_position, position):
    side_reward = - 0.1
    side_reward += 100. * action * velocity # * (10 ** (np.sign(action) != np.sign(velocity)))
    # side_reward += 100. * velocity ** 2
    side_reward += 0.5 * (next_position + 1.2) ** 2
    # side_reward += 0.1 * (abs(next_position - position) * 10) ** 2
    side_reward += 10. * np.logical_and(next_position > 0, position < 0)
    side_reward += 25. * np.logical_and(next_position > 0.2, position < 0.2)
    side_reward += 50. * np.logical_and(next_position > 0.4, position < 0.4)
    return reward + side_reward

Second over 200tsd step and reward reshape
def reshape_reward(reward, action, velocity, next_position, position):
    side_reward = - 0.1
    side_reward += 100. * action * velocity # * (10 ** (np.sign(action) != np.sign(velocity)))
    # side_reward += 100. * velocity ** 2
    side_reward += 0.1 * (next_position + 1.2) ** 2
    # side_reward += 0.1 * (abs(next_position - position) * 10) ** 2
    side_reward += 10. * np.logical_and(next_position > 0, position < 0)
    side_reward += 25. * np.logical_and(next_position > 0.2, position < 0.2)
    side_reward += 50. * np.logical_and(next_position > 0.4, position < 0.4)
    return reward + side_reward

### Example of an episode
[![Watch the video](https://dagshub.com/boguslawski.m.j/Reinforcement-learning/src/main/Mountain-Car/Mountain-Car-example.gif)](https://dagshub.com/boguslawski.m.j/Reinforcement-learning/src/main/Mountain-Car/Mountain-Car-example.mp4)


## Mountain Car PP0
### Problem Description
Model used: Simple PPO with reshaped rewards
Result: 97.3 +- 0.7

### Example of an episode
![Watch the video](https://dagshub.com/boguslawski.m.j/Reinforcement-learning/src/main/Mountain-Car-PPO/Mountain-Car-example.mp4)
