import blockulib
import blockulib.models as blom
import torch
from abc import ABC, abstractmethod
from blockulib.data import DataTransformer
from blockulib.utils import PositionList, ShallowList, DeepList

class PlayingLoop():
    
    def __init__(self, pos_list_type = DeepList):
        self.generator = blockulib.BlockGenerator()
        self.pos_list_type = pos_list_type
    
    def __call__(self, num_games = 1, starting_positions = None, pick_moves_config = {}):
        self.pos_list = self.pos_list_type(num_games, starting_positions)
        self.num_games = self.pos_list.num_games
        self.move = 0
        
        while (self.pos_list.active_games > 0 and self.continue_condition()):
            self.move += 1
            boards = self.pos_list.active_boards()
            pos, ind = blockulib.possible_moves(boards, self.generator)
            chosen_moves = self.pick_moves(self.pos_list.active_games, pos, ind, **pick_moves_config)
            self.pos_list.process_chosen_moves(chosen_moves)
                
        return self.pos_list()
    
    def continue_condition(self,):
        return True
    
    @abstractmethod
    def pick_moves(self, how_many, pos, ind):
        pass
    
class SimpleLoop(PlayingLoop):
    
    def pick_moves(self, how_many, pos, ind):
        chosen_moves = torch.full((how_many, 9, 9), torch.nan)
        for i in range(ind.shape[0]):
            index = int(ind[i].item())
            if chosen_moves[index].isnan().any():
                chosen_moves[index] = pos[i]
        return chosen_moves
    
class ModelBasedLoop(PlayingLoop):
    
    def __init__(self, pos_list_type = DeepList, model_path = "models/conv_model.pth", architecture = blom.ConvModel):
        super().__init__(pos_list_type)
        self.model = architecture()
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def pick_moves(self, how_many, pos, ind, pred_config = {}, rethink_config = {}, temperature = 1.0, top_k: int = 5):
        logits = self.get_model_pred(data = pos, **pred_config)
        pos, ind, logits = blockulib.cut_to_topk(pos, ind, logits, num_games = self.pos_list.active_games, top_k = top_k)
        logits = self.rethink_logits(pos, logits, **rethink_config)
        decisions = blockulib.logits_to_choices(logits, ind, self.pos_list.active_games, temperature = temperature, top_k = top_k)
        
        chosen_moves = torch.full((self.pos_list.active_games, 9, 9), torch.nan)
        for i in range(self.pos_list.active_games):
            if decisions[i] is not None:
                chosen_moves[i] = pos[decisions[i]]
        return chosen_moves
                    
    def get_model_pred(self, data, batch_size = 2048, device = None):
        if (data.shape[0] == 0):
            return torch.tensor([])
        if device is None:
            device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        predictions = []
        
        with torch.no_grad():
            for i in range(0, data.shape[0], batch_size):
                batch = data[i:i+batch_size].to(device)
                batch = batch.unsqueeze(1)
                output = self.model(batch)
                predictions.append(output.cpu())
            
        return torch.cat(predictions).squeeze(1)
    
    def rethink_logits(self, pos, logits):
        return logits
    
class Probe(ModelBasedLoop):
    
    def __init__(self, depth = 5, pos_list_type = ShallowList):
        super().__init__(pos_list_type = pos_list_type)
        self.depth = depth
        
    def continue_condition(self,):
        return self.move < self.depth
    

class DeepSearch(ModelBasedLoop):
    def __init__(self, pos_list_type = DeepList, probe_config = {}, model_path = "models/conv_model.pth", architecture = blom.ConvModel):
        super().__init__(pos_list_type = pos_list_type, model_path = model_path, architecture = architecture)
        self.probe = Probe(pos_list_type = ShallowList, **probe_config)
        self.dt = DataTransformer()
    
    def rethink_logits(self, data, old_logits, rethink_batch = 100, probes_per_pos = 5, probe_config = {}):
        if (data.shape[0] == 0):
            return torch.tensor([])
        
        end_state_list, length_list = [], []
        for i in range(0, data.shape[0], rethink_batch):
            batch = data[i:i+rethink_batch]
            batch = batch.repeat_interleave(probes_per_pos, dim = 0)
            probe_states, probe_lengths = self.probe(starting_positions = batch, **probe_config)
            
            end_state_list.append(probe_states)
            length_list.append(probe_lengths)
        
        all_lengths = torch.cat(length_list)
        all_states = torch.cat(end_state_list)
        needs_pred = torch.where(all_lengths == self.probe.depth)
        
        pred_logits = self.get_model_pred(all_states[needs_pred])
        all_lengths[needs_pred] += self.dt.logits_to_nums(pred_logits)
        average_length = all_lengths.view(data.shape[0], probes_per_pos).mean(dim = 1)
        return self.dt.nums_to_logits(average_length)

from tqdm import tqdm

def play_games(num_games, games_at_once, playing_loop: PlayingLoop, loop_init_config = {}, playing_config = {}, save: bool = False, save_dir = "data/tensors/"):
    x_list, y_list = [], []
    loop = playing_loop(**loop_init_config)
    total_moves = 0
    
    for left in tqdm(range(0, num_games, games_at_once), desc = "Playing games"):
        right = min(num_games, left+games_at_once)
        data = loop(num_games = (right - left), **playing_config)
        for i in range((right-left)):
            total_moves += len(data[i])-1
            y_list.append(torch.linspace(len(data[i])-1, 0, steps = len(data[i])))
            x_list.append(torch.stack(data[i]))
    torch.cuda.empty_cache()
    
    if save:
        list_dict = {'x_list' : x_list, 'y_list' : y_list}
        torch.save(list_dict, save_dir + "lists.pth")
    
    print(f"Mean MPG : {(total_moves/num_games)}")
    return
