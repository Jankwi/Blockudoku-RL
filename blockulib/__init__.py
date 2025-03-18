import pickle, random, torch

class BlockGenerator():
    
    def __init__(self, dict_dir = "data/placements_dict.pkl", to_numpy = False):
        pldict = {}
        self.list = []
        with open(dict_dir, "rb") as f:
            pldict = pickle.load(f)
        self.size = pldict['num_blocks']
        for i in range(self.size):
            self.list.append(pldict[i])
            if to_numpy:
                self.list[i] = self.list[i].numpy()
            
    def __call__(self, num_blocks):
        return [self.list[random.randint(0, self.size-1)].clone() for i in range(num_blocks)]

    

import numpy as np
from numba import guvectorize, float32

@guvectorize([(float32[:,:], float32[:,:])], '(m,n)->(m,n)', target='parallel')
def clear_board(input_square, output_square):
    #Copying values
    for i in range(input_square.shape[0]):
        for j in range(input_square.shape[1]):
            output_square[i, j] = input_square[i, j]
    
    #Erasing rows
    for i in range(input_square.shape[0]):
        suma = 0
        for j in range(input_square.shape[1]):
            suma += input_square[i, j]
        if suma == 9:
            for j in range(input_square.shape[1]):
                output_square[i, j] = 0
    
    #Erasing columns
    for j in range(input_square.shape[0]):
        suma = 0
        for i in range(input_square.shape[1]):
            suma += input_square[i, j]
        if suma == 9:
            for i in range(input_square.shape[1]):
                output_square[i, j] = 0
    
    #Erasing squares
    for i in range(3):
        for j in range(3):
            suma = 0
            for k in range(3):
                for l in range(3):
                    suma += input_square[3*i + k, 3*j + l]
            if suma == 9:
                for k in range(3):
                    for l in range(3):
                        output_square[3*i + k, 3*j + l] = 0
    
    
def possible_moves(boards, generator):
    n = len(boards)
    blocks = generator(len(boards))
    for i in range(n):
        blocks[i] += boards[i]
        
    pos = torch.cat(blocks)
    indices = torch.cat([i*torch.ones(blocks[i].shape[0]) for i in range(n)])
    
    mask = (pos > 1).any(dim = (1, 2))
    pos = pos[~mask].numpy()
    indices = indices[~mask]
    pos = clear_board(pos)
    
    return torch.from_numpy(pos), indices


import torch.nn.functional as F
def sample_from_logits(logits, temperature = 1.0, top_k: int = None):
    logits = logits / temperature

    if top_k is not None and top_k < logits.size(-1):
        top_vals, _ = torch.topk(logits, k=top_k)
        threshold = top_vals[-1]
        logits[logits < threshold] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)
    return sampled_index.item()

def logits_to_choices(logits, index, num_games, temperature = 1.0, top_k: int = None):
    range_left = [None for i in range(num_games)]
    range_right = [None for i in range(num_games)]
        
    for i in range(index.shape[0]):
        ind = int(index[i].item())
        if range_left[ind] is None:
            range_left[ind] = i
        range_right[ind] = i+1
            
    choices = []
    for left, right in zip(range_left, range_right):
        if left is not None and right is not None:
            choices.append(left+sample_from_logits(logits[left:right], temperature = temperature, top_k = top_k))
        else:
            choices.append(None)
    return choices