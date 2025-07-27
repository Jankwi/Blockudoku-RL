import torch
from abc import ABC, abstractmethod

class PositionList():
    @abstractmethod
    def __init__(self, num_games, starting_positions = None):
        pass
        
    @abstractmethod
    def __call__(self,):
        pass
    
    #returns a (num_active, 9, 9) shaped tensor with all active boards
    @abstractmethod
    def active_boards(self,):
        pass
    
    @abstractmethod
    def process_chosen_moves(self, chosen_moves):
        pass

#Keeps a list of lists, tracking each game's sequence of moves
class DeepList(PositionList):
    def __init__(self, num_games, starting_positions = None):
        if starting_positions is None:
            self.pos_list = [[torch.zeros(9, 9)] for i in range(num_games)]
            self.num_games = num_games
        else:
            self.num_games = starting_positions.shape[0]
            self.pos_list = []
            for i in range(self.num_games):
                self.pos_list.append([starting_positions[i]])
                
        self.active_games = self.num_games
        self.state = torch.ones(self.num_games, dtype = torch.bool)
        
    def __call__(self,):
        return self.pos_list
    
    def active_boards(self,):
        self.new_index = torch.where(self.state == True)[0].tolist()
        active_list =  [self.pos_list[self.new_index[i]][-1].clone() for i in range(self.active_games)]
        return torch.stack(active_list)
    
    def process_chosen_moves(self, chosen_moves):
        for i in range(self.active_games):
            if not chosen_moves[i].isnan().any():
                self.pos_list[self.new_index[i]].append(chosen_moves[i].clone())
            else:
                self.state[self.new_index[i]] = False
                self.active_games -= 1
                
#Keeps a tensor, with the latest position for each game - fully vectorized
class ShallowList(PositionList):
    def __init__(self, num_games, starting_positions = None):
        if starting_positions is None:
            self.pos_tensor = torch.zeros((num_games, 9, 9))
            self.num_games = num_games
        else:
            self.pos_tensor = starting_positions.clone()
            self.num_games = starting_positions.shape[0]
            
        self.active_games = self.num_games
        self.game_lengths = torch.zeros(self.num_games)
        self.state = torch.ones(self.num_games, dtype = torch.bool)
        
    def __call__(self,):
        return self.pos_tensor, self.game_lengths
    
    def active_boards(self,):
        self.new_index = torch.where(self.state == True)[0]
        return self.pos_tensor[self.state == True].clone()
    
    def process_chosen_moves(self, chosen_moves):
        with_nan = torch.isnan(chosen_moves).any(dim = (1, 2))
        without_nan = ~with_nan
        
        self.active_games -= with_nan.sum().item()
        actual_with_nan = self.new_index[with_nan]
        actual_without_nan = self.new_index[without_nan]
        
        self.state[actual_with_nan] = False
        self.game_lengths[actual_without_nan] += 1
        self.pos_tensor[actual_without_nan] = chosen_moves[without_nan]