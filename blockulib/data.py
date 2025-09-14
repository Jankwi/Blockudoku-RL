import torch
import pickle
import random
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class DataTransformer():
    
    def __init__(self, dict_dir = "data/tensors/"):
        tr_dict = {}
        with open((dict_dir + "transform_params.pkl"), "rb") as f:
            tr_dict = pickle.load(f)
        self.mean_val = tr_dict['mean']
        self.std_val = tr_dict['std']
        
    def nums_to_logits(self, tensor, eps = 1e-6):
        tensor = torch.log(1 + tensor)
        return (tensor - self.mean_val) / (self.std_val + eps)
    
    def logits_to_nums(self, tensor, eps = 1e-6):
        tensor *= (self.std_val + eps)
        tensor += self.mean_val
        tensor = torch.exp(tensor) - 1
        return torch.clamp(tensor, min = 0)

class DataOrganizer():
    
    def __init__(self, ):
        pass
        
    def __call__(self, iteration = 2137, save_dir = "data/tensors/", prep_data = True, choose_config = {}, show = False, num_bins = None):
        if prep_data:
            self.prep_data(save_dir, choose_config) # optional call
        
        y_dict = torch.load(save_dir + "y.pth")
        y = y_dict['y']
        
        if num_bins is None:
            num_bins = int(y.max() + 1)

        self.diagram(y, bins = num_bins, diagram_name = f"original{iteration//10}{iteration%10}", show = show)
        y = self.transform(y)
        self.diagram(y, bins = num_bins, diagram_name = f"transformed{iteration//10}{iteration%10}", show = show)
        print(y.shape)
                          
        y_dict = {'y' : y}
        torch.save(y_dict, save_dir + "transformed_y.pth")
        
        tr_dict = {'mean' : self.mean_val, 'std' : self.std_val}
        with open(save_dir + "transform_params.pkl", "wb") as f:
            pickle.dump(tr_dict, f)
            
    def prep_data(self, save_dir, choose_config = {}):
        list_dict = torch.load(save_dir + "lists.pth")
        x_list = list_dict['x_list']
        y_list = list_dict['y_list']
        assert(len(x_list) == len(y_list))
        
        x, y = self.choose_boards(x_list = x_list, y_list = y_list, **choose_config)
        
        x_dict = {'x' : x}
        torch.save(x_dict, save_dir + "x.pth")
        y_dict = {'y' : y}
        torch.save(y_dict, save_dir + "y.pth")
            
    def choose_boards(self, x_list, y_list):
        x = torch.cat(x_list)
        y = torch.cat(y_list)
        assert(x.shape[0] == y.shape[0])
        return x, y
        
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
        self.mean_val = tensor.mean()
        self.std_val = tensor.std()
        return (tensor - self.mean_val) / (self.std_val + eps)
    
class OneFromEach(DataOrganizer):
    
    def choose_boards(self, x_list, y_list):
        chosen_x = []
        chosen_y = []
        
        for i in range(len(x_list)):
            chosen_ind  = random.randint(0, len(x_list[i])-1)
            chosen_x.append(x_list[i][chosen_ind])
            chosen_y.append(y_list[i][chosen_ind])
                                         
        x = torch.stack(chosen_x)
        y = torch.stack(chosen_y)
        return x, y
        
class RandomBoards(DataOrganizer):
    
    def choose_boards(self, x_list, y_list, n = None, multiplier = 1.0, cap = None):
        x = torch.cat(x_list)
        y = torch.cat(y_list)
        assert(x.shape[0] == y.shape[0])
        
        if n is None:
            n = int(multiplier * len(x_list))
        if cap is not None:
            n = min(n, cap)
            
        chosen_idx = torch.randperm(x.shape[0])[:n]
        x = x[chosen_idx]
        y = y[chosen_idx]
        return x, y
          
def get_unique(tensor):
    k = tensor.shape[1]
    flat = tensor.view(-1, k*k)

    unique_flat, inverse, counts = torch.unique(
        flat, dim=0, return_inverse=True, return_counts=True
    )
    unique_blocks = unique_flat.view(-1, k, k)

    return unique_blocks, inverse, counts
    
class MostPopular(DataOrganizer):
        
    def choose_boards(self, x_list, y_list, threshold = 3):
        all_xs = torch.cat(x_list, dim = 0)
        unique_blocks, inverse, counts = get_unique(all_xs)
        relevant = torch.where(counts >= threshold)[0]
        
        n_relevant = relevant.shape[0]
        new_ys = torch.zeros(n_relevant)
        all_ys   = torch.cat(y_list,  dim=0)

        for i in range(n_relevant):
            occurences = (inverse == relevant[i])
            new_ys[i] = all_ys[occurences].mean().item()
            
        return unique_blocks[relevant], new_ys
    
    
class XTransform():
    def __init__(self, ):
        pass
    
    def __call__(self, save_dir = "data/tensors/", transform_config = {}):
        list_dict = torch.load(save_dir + "lists.pth")
        x_list = list_dict['x_list']
        y_list = list_dict['y_list']
        assert(len(x_list) == len(y_list))
        
        x = self.transform(x_list = x_list, y_list = y_list, **transform_config)
        x_dict = {'x' : x}
        torch.save(x_dict, save_dir + "x.pth")
    
    @abstractmethod
    def transform(self, x_list, y_list):
        pass
    
class NMostPopular(XTransform):
    
    def transform(self, x_list, y_list, top_n):
        all_xs = torch.cat(x_list, dim = 0)
        unique_blocks, inverse, counts = get_unique(all_xs)
        
        print(f"{top_n} boards requested, out of {unique_blocks.shape[0]}")
        if unique_blocks.shape[0] < top_n:
            print("Warning - not enough boards, lowering top_n")
            top_n = unique_blocks.shape[0]
        
        threshold, prioritised = 1, (counts >= 1)
        while prioritised.sum().item() > top_n:
            threshold += 1
            prioritised = (counts >= threshold)
        transformed = unique_blocks[prioritised]    
        
        places_left = top_n - prioritised.sum().item()
        print(f"Threshold set at {threshold}, {places_left} places left")
        if places_left > 0:
            candidates = unique_blocks[counts == (threshold - 1)]
            chosen_idx = torch.randperm(candidates.shape[0])[:places_left]
            transformed = torch.cat([transformed, candidates[chosen_idx]])
            
        return transformed

class YDiscounter():
    def __init__(self, dsc_rate = 0.95):
        self.q = torch.as_tensor(dsc_rate)
        
    def lengths_to_values(self, lengths : torch.Tensor):
        if (self.q.item() == 1):
            return length
        return (1 - torch.pow(self.q, lengths)) / (1 - self.q)