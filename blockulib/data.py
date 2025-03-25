import torch
import pickle
import matplotlib.pyplot as plt

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
        
        tr_dict = {'mean' : self.mean_val, 'std' : self.std_val}
        with open(save_dir + "transform_params.pkl", "wb") as f:
            pickle.dump(tr_dict, f)
        
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