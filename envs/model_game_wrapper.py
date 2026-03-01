from pathlib import Path
import sys
import numpy as np

parent_path = Path(__file__).parent.parent.parent
sys.path.append(str(parent_path))

from solitaire_game.game import Game
from solitaire_game.game_blocks import Tableau, Foundation, Waste, Stock
from solitaire_game.utils import Card


class GymLikeGameWrapper(Game):
    def __init__(
        self, 
        verbose: bool = False, 
        max_iter: int = 200,
        move_penalty: float = 0,
        truncation_penalty: float = 0,
        win_reward: float = 1e6,
    ):
        super().__init__(verbose=verbose)
        self.state = np.zeros(shape=(10, 1), dtype=np.int16)
        self.max_iter = max_iter
        self.curr_iter = 0
        
        # to calculate reward
        self.move_penalty = move_penalty
        self.truncation_penalty = truncation_penalty
        self.win_reward = win_reward
    
    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.curr_iter = 0
        info = {}
        state = self._convert_to_numpy()
        
        return state, info
    
    def step(self, move_id: int):
        pile_from = move_id // 10
        pile_to = move_id % 10
        result = super().move(pile_from=pile_from, pile_to=pile_to)
        terminated = np.array([self.is_win(), ])
        self.curr_iter += 1
        truncated = np.array([( self.curr_iter >= self.max_iter )])
        reward = self._reward_function(
            result=result, 
            pile_from=pile_from, 
            pile_to=pile_to, 
            terminated=terminated, 
            truncated=truncated,
        )
        state = self._convert_to_numpy()
        info = {}
        return state, reward, terminated, truncated, info
    
    @staticmethod
    def __get_card_id(card: Card) -> int:
        return card.color * 13 + card.figure + 2
    
    def _reward_function(
        self,
        result: bool, 
        pile_from: int, 
        pile_to: int, 
        terminated: np.ndarray, 
        truncated: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        reward = -self.move_penalty
        if result:
            reward += self.__move_reward(pile_from=pile_from, pile_to=pile_to)
        
        if terminated:
            reward += self.win_reward
            
        if truncated:
            reward -= self.truncation_penalty
        
        return np.array([reward, ])
    
    @staticmethod
    def __move_reward(pile_from: int, pile_to: int) -> float:
        reward = 0.
        if pile_to == 9:
            reward += 0.2
        elif pile_from == 9:
            reward -= 0.15
        elif pile_from == 7 and pile_to == 8:
            reward += 0.01
        else:
            reward += 0.15
        return reward
        
    def _convert_to_numpy(self):
        tableau_state = self.__convert_tableau_to_numpy(self.tableau)
        foundation_state = self.__convert_foundation_to_numpy(self.foundation)
        waste_state = self.__convert_waste_to_numpy(self.waste)
        stock_state = self.__convert_stock_to_numpy(self.stock)
        return (tableau_state, foundation_state, waste_state, stock_state)
    
    @staticmethod
    def __convert_tableau_to_numpy(tableau: Tableau) -> np.ndarray:
        temp_array = np.zeros(shape=(15, 7), dtype=np.int16)
        for col, (faceup, pile) in enumerate(zip(tableau.faceup, tableau.piles)):
            for row, card in enumerate(pile):
                if row >= faceup:
                    result = GymLikeGameWrapper.__get_card_id(card)
                else:
                    result = 1
                temp_array[row, col] = result
        return temp_array
    
    @staticmethod
    def __convert_foundation_to_numpy(foundation: Foundation) -> np.ndarray:
        temp_array = np.zeros(shape=(4, ), dtype=np.int16)
        for id, pile in enumerate(foundation.foundation):
            if len(pile) > 0:
                temp_array[id] = GymLikeGameWrapper.__get_card_id(pile[-1])
        return temp_array
    
    @staticmethod
    def __convert_waste_to_numpy(waste: Waste):
        temp_array = np.zeros(shape=(1, ), dtype=np.int16)
        card = waste.get()
        if card:
            temp_array[0] = GymLikeGameWrapper.__get_card_id(card)
        return temp_array
    
    @staticmethod
    def __convert_stock_to_numpy(stock: Stock):
        temp_array = np.zeros(shape=(1, ), dtype=np.int16)
        temp_array[0] = len(stock)
        return temp_array
    
if __name__ == "__main__":
    game = GymLikeGameWrapper()
    output, _ = game.reset()
    print(output[0].shape)
    print(game)
