import blockulib
import blockulib.models as blom
import torch

class PlayingLoop():
    
    def __init__(self, model_path = "models/conv_model.pth", architecture = blom.ConvModel):
        self.model = architecture()
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.generator = blockulib.BlockGenerator()
        self.model.eval()
        
    def __call__(self, num_games = 1, batch_size = 4096, temperature = 1.0, top_k: int = None):
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
            logits = self.get_model_pred(pos, batch_size = batch_size).squeeze(1)
            decisions = blockulib.logits_to_choices(logits, ind, active_games, temperature = temperature, top_k = top_k)
            
            for i in range(active_games):
                if (decisions[i] is None):
                    state[new_index[i]] = False
                    active_games -= 1
                else:
                    pos_list[new_index[i]].append(pos[decisions[i]])
                    
        print("ended after ", move, " moves")            
        return pos_list
                    
    def get_model_pred(self, data, batch_size, device = None):
        if (data.shape[0] == 0):
            return torch.tensor([[]])
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
            
        return torch.cat(predictions)