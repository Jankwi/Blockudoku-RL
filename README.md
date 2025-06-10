# Blockudoku RL
Still a work in progress, the Blockudoku-RL repository is home to the blockulib - a Reinforcement Learning library for automating the game of Blockudoku - a Sudoku and tetris crossover.
This repository also contains .ipynb notebooks mostly serving as working spaces or user examples (often times both ;->)

![The rules of Blockuodoku](https://easybrain.com/uploads/media/1920x1080/08/108-New%20Easybrain%20puzzle%20BlockuDoku%20is%20now%20available%20on%20the%20App%20Store%20worldwide.%20.jpg?v=1-0)

## Overview

The main pipeline enabled by the library, consists of a loop with the following steps:
* playing (data collection) - collecting different blockudoku boards, and their expected moves value

  (either at random or using a trained model)
* data transformation - using data transformation/selection techniques to prepare the data for training
* model training - training a model, whcih predicts the expected moves value of a blockudoku board

Very basic pipelines like this are already implemented in the library.  
However, more sophisticated methods have to be implemented for the model to achieve any bigger results.   
therefore the library is still a work in progress.
## Library structure:
The blockulib currently consists of these modules:

### __init__
The base library consists of methods, which are strictly connected to the game and it's vectorised representation:
* `class BlockGenerator` - randomly chooses a block and generates a list of 9x9 boards with all of the block's possible configurations on the board
* `def clear_board(input_square, output_square)` - a numba vectorized method, which performs the board cleaning of a tensor containing blockudoku boards
* `def possible_moves(boards, generator):` - this methods requires a little more attention:

  $\underline{input:}$ $boards$ - a (m, 9, 9) shaped tensor with each of the $m$ blockudoku boards containing a valid state of play, $generator$ - a $BlockGenerator$ istance
  
  $\underline{output:}$ A (S, 9, 9) shaped tensor with all the possible moves for each of $boards$'s positions concatenated into one (Let $p_i$ be the $i$-th element of boards and $s(p_i)$ is the numer of possible moves from position $p_i$, then $S = \sum_{i = 1}^{m} s(p_i)$. Another tensor $indices$ is also returned to keep track which possible move came from which position.

There are also some ML helper functions inside:
* `def sample_from_logits(logits, temperature = 1.0, top_k: int = None):`
* `def indices_to_range(num_games, index):`
* `def cut_to_topk(pos, index, logits, num_games, top_k):`
* `def logits_to_choices(logits, index, num_games, temperature = 1.0, top_k: int = None):`

### data (more info soon)
All methods and classess connected to processing the data

### models (more soon)
A module containing the declarations of custom (torch.nn.Module inherent) models.

### pipelines (more info soon)
The pipelines module is home to the `Configs` class, which contains many hekpful cofigs.  
The `Pipeline' class, enables executing a full RL loop many times.

### playing (more info soon)
The playing module contains many classess and methods, useful in collecting data about possible blockudoku positions.

### training
The data module contains classess and methods useful for training a model on the prieviously collected data.
