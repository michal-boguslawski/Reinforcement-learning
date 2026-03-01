import pytest
import numpy as np

from ...memory.replay_buffer import ReplayBuffer


def generate_replay_buffer_with_data_simple_states(
    buffer: ReplayBuffer,
    samples_size: int, 
    state_shape: tuple = (), 
    action_shape: tuple = (),
) -> ReplayBuffer:
    for _ in range(samples_size):
        state = np.random.random(state_shape)
        action = np.random.randint(0, 9, size=action_shape)
        reward = np.random.random()
        done = np.random.choice([True, False])
        buffer.push((state, action, reward, done))
    return buffer

def test_replay_buffer_one_timestep_shapes():
    buffer = ReplayBuffer(buffer_size=100)
    buffer = generate_replay_buffer_with_data_simple_states(
        buffer=buffer,
        samples_size=50,
        state_shape=(4),
        action_shape=(1)
    )
    
    batch_size = 16
    sample = buffer.sample(batch_size)
    
    assert sample[0].shape == (batch_size, 4)
    assert sample[1].shape == (batch_size, 1)
    assert sample[2].shape == (batch_size, )
    assert sample[3].shape == (batch_size, )

def test_replay_buffer_item_vanish_more_push_than_size():
    buffer = ReplayBuffer(buffer_size=10)
    first_item = (np.random.random(4))
    buffer.push(first_item)
    buffer = generate_replay_buffer_with_data_simple_states(
        buffer=buffer,
        samples_size=10,
        state_shape=(4),
        action_shape=(1)
    )    
    assert id(first_item) not in [id(item) for item in buffer.buffer]
