import pytest
import torch as T

from ...utils.utils import step_return_discounting

def test_simple_rewards_discounting_from_rewards_value():
    discount = 0.99
    rewards = T.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5]
        ]
    )
    next_values = T.tensor([3, -8])
    dones = T.tensor(
        [
            [False, False, False, False],
            [False, False, True, False]
        ]
    )
    
    expected_results = T.tensor(
        [
            [
                1 * (discount ** 0) + 2 * (discount ** 1) + 3 * (discount ** 2) + 4 * (discount ** 3) + 3 * (discount ** 4), 
                2 * discount ** 0 + 3 * discount ** 1 + 4 * discount ** 2 + 3 * discount ** 3,
                3 * discount ** 0 + 4 * discount ** 1 + 3 * discount ** 2,
                4 * discount ** 0 + 3 * discount ** 1,
            ],
            [
                2 * discount ** 0 + 3 * discount ** 1 + 4 * discount ** 2,
                3 * discount ** 0 + 4 * discount ** 1,
                4 * discount ** 0,
                5 * discount ** 0 - 8 * discount ** 1
            ]
        ]
    )
    
    results = step_return_discounting(
        states=rewards,
        next_values=next_values,
        dones=dones,
        discount=discount
    )
    
    assert T.allclose(expected_results, results, rtol=1e-5, atol=1e-8)