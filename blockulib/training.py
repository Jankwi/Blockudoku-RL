import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import blockulib.models as blom

class Trainer():
    
    def __init__(self, architecture = blom.ConvModel, tensor_dir = "data/tensors/"):
        self.fetch_tensors(tensor_dir = tensor_dir)
        self.model = architecture()
        
    def save(self, save_path = "models/conv_model.pth"):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), save_path)
    
    def __call__(self, num_epochs, step_size, batch_size = 128, log_every = 1, save_best = True):
        best_so_far = 10000.0
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
        print("Shape ", x.shape)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        
def train_model(Train: Trainer = Trainer, train_init_config = {}, train_config = {}, train_save_config = {}):
    train = Train(**train_init_config)
    train(**train_config)
    train.save(**train_save_config)