import blockulib
import blockulib.models as blom
import matplotlib.pyplot as plt
import torch
from abc import ABC, abstractmethod
from blockulib.data import DataTransformer

class PlayingLoop():
    @abstractmethod
    def __call__():
        "Run the playing loop"
        pass

class SimpleLoop(PlayingLoop):
    
    def __init__(self):
        self.generator = blockulib.BlockGenerator()
        
    def __call__(self, num_games = 1, batch_size = 128, temperature = None, top_k = None):
        pos_list = [[torch.zeros(9, 9)] for i in range(num_games)]
        state = [True for i in range(num_games)]
        active_games = num_games
        move = 0
        
        while (active_games > 0):
            move += 1
            new_index = []
            for i in range(num_games):
                if state[i]:
                    new_index.append(i)
            boards = [pos_list[new_index[i]][-1].clone() for i in range(active_games)]
            pos, ind = blockulib.possible_moves(boards, self.generator)
            
            state = [False for i in range(num_games)]
            active_games = 0
            for i in range(ind.shape[0]):
                index = new_index[int(ind[i].item())]
                if (not state[index]):
                    state[index] = True
                    pos_list[index].append(pos[i])
                    active_games +=1
        
        return pos_list

class ModelBasedLoop(PlayingLoop):
    
    def __init__(self, model_path = "models/conv_model.pth", architecture = blom.ConvModel):
        self.model = architecture()
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.generator = blockulib.BlockGenerator()
        self.model.eval()
        
    def __call__(self, num_games = 1, rethink_batch = 400, batch_size = 4096, temperature = 1.0, top_k: int = 5):
        pos_list = [[torch.zeros(9, 9)] for i in range(num_games)]
        state = [True for i in range(num_games)]
        active_games = num_games
        move = 0
        
        while (active_games > 0):
            move += 1
            new_index = []
            for i in range(num_games):
                if state[i]:
                    new_index.append(i)
            boards = [pos_list[new_index[i]][-1].clone() for i in range(active_games)]
            
            pos, ind = blockulib.possible_moves(boards, self.generator)
            logits = self.get_model_pred(pos, batch_size = batch_size)
            pos, ind, logits = blockulib.cut_to_topk(pos, ind, logits, num_games = active_games, top_k = top_k)
            logits = self.rethink_logits(pos, logits, rethink_batch)
            decisions = blockulib.logits_to_choices(logits, ind, active_games, temperature = temperature, top_k = top_k)
            
            for i in range(active_games):
                if (decisions[i] is None):
                    state[new_index[i]] = False
                    active_games -= 1
                else:
                    pos_list[new_index[i]].append(pos[decisions[i]])
                    
        #print("ended after ", move, " moves")            
        return pos_list
                    
    def get_model_pred(self, data, batch_size, device = None):
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
    
    def rethink_logits(self, pos, logits, rethink_batch):
        return logits


class Probe(ModelBasedLoop):
    
    def __call__(self, pos_tensor, depth = 5, batch_size = 4096, temperature = 1.0, top_k: int = None):
        num_games = pos_tensor.shape[0]
        state = [True for i in range(num_games)]
        game_length = torch.zeros(num_games)
        last_logit = torch.full((num_games, ), float('-inf'))
        active_games = num_games
        
        for d in range(depth):
            new_index = []
            for i in range(num_games):
                if state[i]:
                    new_index.append(i)
            boards = [pos_tensor[new_index[i]].clone() for i in range(active_games)]
            pos, ind = blockulib.possible_moves(boards, self.generator)
            logits = self.get_model_pred(pos, batch_size = batch_size)
            decisions = blockulib.logits_to_choices(logits, ind, active_games, temperature = temperature, top_k = top_k)
            
            for i in range(active_games):
                if (decisions[i] is None):
                    state[new_index[i]] = False
                    active_games -= 1
                else:
                    pos_tensor[new_index[i]] = pos[decisions[i]]
                    game_length[new_index[i]] += 1.
                    last_logit[new_index[i]] = logits[decisions[i]]
            if (active_games == 0):
                break
           
        return game_length, last_logit

class DeepSearch(ModelBasedLoop):
    def __init__(self):
        super().__init__()
        self.probe = Probe()
        self.dt = DataTransformer()
    
    def rethink_logits(self, data, old_logits, rethink_batch, depth = 3, probes_per_pos = 5):
        if (data.shape[0] == 0):
            return torch.tensor([])
        gls, lls = [], []
        for i in range(0, data.shape[0], rethink_batch):
            batch = data[i:i+rethink_batch]
            batch = batch.repeat_interleave(probes_per_pos, dim = 0)
            gl, ll = self.probe(pos_tensor = batch, depth = depth, temperature = 0.7, top_k = 3)
            gls.append(gl)
            lls.append(ll)
        GLs = torch.cat(gls)
        LLs = torch.cat(lls)
        LLs[torch.where(GLs < depth)] = -10
        GLs += self.dt.logits_to_nums(LLs)
        LLs = self.dt.nums_to_logits(GLs)
        LLs = LLs.view(data.shape[0], probes_per_pos).mean(dim = 1)
        return LLs

from tqdm import tqdm

def play_games(num_games, games_at_once, playing_loop: PlayingLoop, save: bool = False, save_dir = "data/tensors/", batch_size = 4096, temperature = 1.0, top_k: int = None):
    x_list, y_list = [], []
    loop = playing_loop()
    
    for left in tqdm(range(0, num_games, games_at_once), desc = "Playing games"):
        right = min(num_games, left+games_at_once)
        data = loop(num_games = (right - left), batch_size = batch_size, temperature = temperature, top_k = top_k)
        for i in range((right-left)):
            y_list.append(torch.linspace(len(data[i])-1, 0, steps = len(data[i])))
            x_list.append(torch.stack(data[i]))
    torch.cuda.empty_cache()
            
    x = torch.cat(x_list)
    y = torch.cat(y_list)
    
    if save:
        x_dict = {'x' : x}
        torch.save(x_dict, save_dir + "x.pth")
        y_dict = {'y' : y}
        torch.save(y_dict, save_dir + "y.pth")
    
    print(f"Mean MPG : {(y.shape[0]/num_games) - 1}")
    return
