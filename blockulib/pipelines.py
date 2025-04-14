import time
import blockulib.models as blom
from blockulib.playing import ModelBasedLoop, DeepSearch
from blockulib.data import DataOrganizer
from blockulib.playing import play_games
from blockulib.training import train_model


class Configs:
    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} is a static utility class and cannot be instantiated.")

    @classmethod
    def simple_playing_config(cls, num_games = 1000, games_at_once = 1000, save = True):
        simple_config = {
            'num_games' : num_games,
            'games_at_once' : games_at_once,
            'playing_loop' : ModelBasedLoop,
            'save' : save,
            'playing_config': {
                'temperature' : 0.7,
                'top_k' : 3,
                'pred_config' : {
                    'batch_size' : 4096
                }
            }
        }
        return simple_config
    
    @classmethod
    def deep_playing_config(cls, num_games = 30, games_at_once = 30, save = True):
        deep_config = {
            'num_games' : num_games,
            'games_at_once' : games_at_once,
            'playing_loop' : DeepSearch,
            'loop_init_config' : {
                'probe_config' : {
                    'model_path' : "models/conv_model.pth",
                    'architecture' : blom.ConvModel
                }
            },
            'save' : save,
            'playing_config': {
                'temperature' : 0.7,
                'top_k' : 3,
                'pred_config' : {
                    'batch_size' : 4096
                },
                'rethink_config' : {
                    'rethink_batch' : 500,
                    'probes_per_pos' : 10,
                    'depth' : 5,
                    'probe_config' : {
                        'batch_size' : 4096,
                        'temperature' : 0.7,
                        'top_k' : 3
                    }
                }
            }
        }
        return deep_config
        
    @classmethod
    def dorg_config(cls):
        return {}
    
    @classmethod
    def dorg_random_config(cls, multiplier = 3., cap = 100*1000):
        dorg_config = {
            'choose_config' : {
                'multiplier' : multiplier,
                'cap' : cap
            }
        }
        return dorg_config
    
    @classmethod
    def train_init_config(cls, architecture = blom.ConvModel):
        train_init_config = {
            'architecture' : architecture,
            'tensor_dir' : "data/tensors/"
        }
        return train_init_config
    
    @classmethod
    def train_config(cls, num_epochs = 5, log_every = 1):
        train_config = {
            'num_epochs' : num_epochs,
            'step_size' : 0.001,
            'batch_size' : 512,
            'log_every' : log_every
        }
        return train_config
    
    @classmethod
    def train_save_config(cls, save_path = "models/conv_model.pth"):
        train_save_config = {
            'save_path' : save_path
        }
        return train_save_config
    
    @classmethod
    def train_model_config(cls, num_epochs = 20, log_every = 5, architecture = blom.ConvModel, save_path = "models/conv_model.pth"):
        train_init_config = Configs.train_init_config(architecture = blom.ConvModel)
        train_config = Configs.train_config(num_epochs = num_epochs, log_every = log_every)
        train_save_config = Configs.train_save_config(save_path = save_path)
        train_model_config = {
            'train_init_config' : train_init_config,
            'train_config' : train_config,
            'train_save_config' : train_save_config
        }
        return train_model_config

class Pipeline():
    
    def __init__(self):
        pass
    
    def __call__(self, num_iterations, play_games_config, train_model_config, dorg_config, start_iteration = 0, Dorg: DataOrganizer = DataOrganizer):
        
        dorg = Dorg()
        for i in range(start_iteration, start_iteration + num_iterations):
            print("Starting ieration ", i)
            start_time = time.time()
    
            play_games(**play_games_config)
            dorg(iteration = i, **dorg_config)
            train_model(**train_model_config)
    
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.6f} seconds")