# Library structure:
The blockulib currently consists of these modules:

## blockulib
The base library consists of methods, which are strictly connected to the game and it's vectorised representation:
* `class BlockGenerator` - randomly chooses a block and generates a list of 9x9 boards with all of the block's possible configurations on the board
* `def clear_board(input_square, output_square)` - a numba vectorized method, which performs the board cleaning of a tensor containing blockudoku boards
* `def possible_moves(boards, generator):` - this methods requires a little more attention:

  $\underline{input:}$ $boards$ - a (m, 9, 9) shaped tensor with each of the $m$ blockudoku boards containing a valid state of play, $generator$ - a $BlockGenerator$ istance
  
  $\underline{output:}$ A (S, 9, 9) shaped tensor with all the possible moves for each of $boards$'s positions concatenated into one 3D tensor. (Let $p_i$ be the $i$-th element of boards and $s(p_i)$ is the numer of possible moves from position $p_i$, then $S = \sum_{i = 1}^{m} s(p_i)$). Another tensor $indices$ is also returned to keep track which possible move came from which position.

There are also some ML helper functions inside:
* `def sample_from_logits(logits, temperature = 1.0, top_k: int = None):`
* `def indices_to_range(num_games, index):`
* `def cut_to_topk(pos, index, logits, num_games, top_k):`
* `def logits_to_choices(logits, index, num_games, temperature = 1.0, top_k: int = None):`

## blockulib.data (more info soon)
All methods and classess connected to processing the data

## blockulib.models (more soon)
A module containing the declarations of custom (torch.nn.Module inherent) models.

## blockulib.pipelines (more info soon)
The pipelines module is home to the `Configs` class, which contains many hekpful cofigs.  
The `Pipeline' class, enables executing a full RL loop many times.

## blockulib.playing (more info soon)
The playing module contains many classess and methods, useful in collecting data about possible blockudoku positions.

## blockulib.training (more info soon)
The data module contains classess and methods useful for training a model on the prieviously collected data.

## blockulib.utils (more info soon)
The utils sub-library contains useful data structures, helpful in modelling multiple blockudoku games at once.