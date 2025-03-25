import blockulib
import blockulib.models as blom
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from abc import ABC, abstractmethod

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

"""
class ModelBasedLoop(PlayingLoop):
    
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
                    
        #print("ended after ", move, " moves")            
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
"""
    
def play_games(num_games, batch_size, playing_loop: PlayingLoop, save_dir = "data/tensors/", temperature = 1.0, top_k: int = None):
    x_list, y_list = [], []
    loop = playing_loop()

    
    for left in range(0, num_games, batch_size):
        right = min(num_games-1, left+batch_size)
        data = loop(num_games = (right - left), temperature = temperature, top_k = top_k)
        for i in range((right-left)):
            y_list.append(torch.linspace(len(data[i])-1, 0, steps = len(data[i])))
            x_list.append(torch.stack(data[i]))
    torch.cuda.empty_cache()
            
    x = torch.cat(x_list)
    x_dict = {'x' : x}
    torch.save(x_dict, save_dir + "x.pth")
    
    y = torch.cat(y_list)
    y_dict = {'y' : y}
    torch.save(y_dict, save_dir + "y.pth")
    
    print(f"Mean MPG : {(y.shape[0]/num_games) - 1}")
    return

class DataOrganizer():
    
    def __init__(self, ):
        pass
        
    def __call__(self, iteration = 2137, save_dir = "data/tensors/", show = False):
        y_dict = torch.load(save_dir + "y.pth")
        y = y_dict['y']
        num_bins = int(y.max() + 1)
        
        self.diagram(y, bins = num_bins, diagram_name = f"original{iteration//10}{iteration%10}", show = show)
        y = self.transform(y)
        self.diagram(y, bins = num_bins, diagram_name = f"transformed{iteration//10}{iteration%10}", show = show)
        print(y.shape)
                          
        y_dict = {'y' : y}
        torch.save(y_dict, save_dir + "transformed_y.pth")
        
        
    def diagram(self, values, bins, save_dir = "data/diagrams/", show = False, diagram_name = ""):
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
        plt.title(diagram_name)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        if show:
            plt.show()
        plt.savefig(save_dir + diagram_name)
        plt.close()
        
        
    def transform(self, tensor, eps = 1e-6):
        tensor = torch.log(1 + tensor)
        mean_val = tensor.mean()
        std_val = tensor.std()
        return (tensor - mean_val) / (std_val + eps)
    
class Trainer():
    
    def __init__(self, architecture = blom.ConvModel, tensor_dir = "data/tensors/"):
        self.fetch_tensors(tensor_dir = tensor_dir)
        self.model = architecture()
        
    def save(self, save_path = "models/conv_model.pth"):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), save_path)
    
    def __call__(self, num_epochs, step_size, batch_size = 128, log_every = 1):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=step_size)
        loss_fn = nn.MSELoss()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        self.model.train()
        self.model.to(device)
        for epoch in range(num_epochs):
            total_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device).unsqueeze(1)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = loss_fn(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            
            if ((epoch+1)%log_every == 0 or epoch == 0):
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
                
                self.model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for x_batch, y_batch in test_loader:
                        x_batch = x_batch.to(device).unsqueeze(1)
                        y_batch = y_batch.to(device)
                        outputs = self.model(x_batch)
                        loss = loss_fn(outputs.squeeze(), y_batch)
                        test_loss += loss.item() * x_batch.size(0)
                    test_loss /= len(test_loader.dataset)
                    print(f"Test Loss: {test_loss:.4f}")
            
        
    def fetch_tensors(self, tensor_dir):
        x_dict = torch.load(tensor_dir + "x.pth")
        y_dict = torch.load(tensor_dir + "transformed_y.pth")
        x = x_dict['x']
        y = y_dict['y']
        dataset = TensorDataset(x, y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])