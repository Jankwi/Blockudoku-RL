import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import blockulib.models as blom

class Trainer():
    
    def __init__(self, architecture = blom.ConvModel, tensor_dir = "data/tensors/"):
        print(f"Initiated Trainer for architecture = {architecture}")
        self.fetch_tensors(tensor_dir = tensor_dir)
        self.model = architecture()
        self.device = 'cpu'
        
    def save(self, save_path = "models/conv_model.pth", final_save = False):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), save_path)
        if not final_save:
            self.model.to(self.device)
        print("Succesfully saved model")
    
    def __call__(self, num_epochs, step_size, batch_size = 128, log_every = 1, save_config = {}, save_best = True):
        self.best_so_far = 10000.0
        self.test_loss = None
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=step_size)
        self.loss_fn = nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        self.model.to(self.device)
        for epoch in tqdm(range(num_epochs), desc = "Epoch"):
            self.model.train()
            total_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device).unsqueeze(1)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.loss_fn(outputs.squeeze(-1), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
                
            valid_loss = self.evaluate_on_dataloader(valid_loader)
            if valid_loss < self.best_so_far:
                self.best_so_far = valid_loss
                self.test_loss = self.evaluate_on_dataloader(test_loader)
                if save_best:
                    self.save(**save_config)
                    
            if ((epoch+1)%log_every == 0):
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Valid Loss: {valid_loss:.4f}")
                    
        self.model.to('cpu')
        print(f"Ended training with the best valid loss of {self.best_so_far} and test loss of: {self.test_loss}")
        return self.test_loss, self.best_so_far
            
    
    def evaluate_on_dataloader(self, dataloader):
        self.model.eval()
        avg_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device).unsqueeze(1)
                y_batch = y_batch.to(self.device)
                outputs = self.model(x_batch)
                loss = self.loss_fn(outputs.squeeze(-1), y_batch)
                avg_loss += loss.item() * x_batch.size(0)
            avg_loss /= len(dataloader.dataset)
        return avg_loss
    
    def fetch_tensors(self, tensor_dir):
        x_dict = torch.load(tensor_dir + "x.pth")
        y_dict = torch.load(tensor_dir + "transformed_y.pth")
        x = x_dict['x']
        y = y_dict['y']
        dataset = TensorDataset(x, y)

        train_valid_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_valid_size
        train_valid_dataset, self.test_dataset = random_split(dataset, [train_valid_size, test_size])
        
        train_size = int(0.9 * len(train_valid_dataset))
        valid_size = len(train_valid_dataset) - train_size
        self.train_dataset, self.valid_dataset = random_split(train_valid_dataset, [train_size, valid_size])
        
        
def train_model(Train: Trainer = Trainer, train_init_config = {}, train_config = {}, train_save_config = {}):
    train = Train(**train_init_config)
    return train(save_config = train_save_config, **train_config)


class ModelGridSearch():
    def __init__(self, architecture_list, num_epochs = 50, batch_size_list = [128, 256, 512], tensor_dir = "data/tensors/", min_lr = 0.001, max_lr = 0.02):
        self.init_config_list = []
        for architecture in architecture_list:
            architecture_config = {
                'architecture' : architecture,
                'tensor_dir' : tensor_dir
            }
            self.init_config_list.append(architecture_config)
        
        self.base_train_config = {
            'num_epochs' : num_epochs,
            'log_every' : 10000,
            'save_best' : False
        }
        self.batch_size_list = batch_size_list
        self.min_lr = 0.001
        self.max_lr = max_lr = 0.02
        
    def objective(self, trial):
        init_config_index = trial.suggest_int("init_config_index", 0, len(self.init_config_list) - 1)
        train_init_config = self.init_config_list[init_config_index]
        trial.set_user_attr("train_init_config", train_init_config)
        
        lr = trial.suggest_float("lr", self.min_lr, self.max_lr, log = True)
        batch_size = trial.suggest_categorical("batch_size", self.batch_size_list)
        
        train_config = self.base_train_config.copy()
        train_config['step_size'] = lr
        train_config['batch_size'] = batch_size
        trial.set_user_attr("train_config", train_config)
        
        test_loss, valid_loss = train_model(train_init_config = train_init_config, train_config = train_config)
        trial.set_user_attr("test_loss", test_loss)
        return valid_loss