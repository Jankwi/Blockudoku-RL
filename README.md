# Blockudoku RL
The Blockudoku-RL repository is home to the blockulib - a Reinforcement Learning library, created from scratch in PyTorch, for automating the game of Blockudoku - a Sudoku and Tetris crossover. The repository also contains usage examples and experiments using the library.  

Promising results have been achieved already, with the library and experiments still in active development.

## Repository structure
The repository consists of:
* The ```blockulib``` directory:  
Library providing useful functions, which help in creating RL experiments and pipelines for the game of blockudoku.
* The ```data``` directory:  
Directory mainly used during experiments for storing datasets/diagrams required to simulate the game.
* ```experiment.ipynb```  
A succesful usage example of the blockulib, to train AI models using ***Reinforcement Learning***.  
The model ***improved from 15.45*** Moves Per Game, when playing at random ***to 160.48***.
* ```visualization.ipynb```  
Python notebook containing visualizations of blockudoku games using different strategies.  

Many blockulib functions also create/reference a ```models``` directory by default.
## The game of Blockudoku
The game of blockudoku, requires the player to place blocks consisting of squares of different shapes on a 9x9 board. After each turn, if any of the 9 rows, 9 columns or 9 sudoku squares is fully covered by blocks, that part of the board is then cleared. The player is served batches of blocks to place on the board, until they can no longer make a move.

The goal of the game is to amass as many points as possible, before the game is over (no move can be made).

![The rules of Blockuodoku](https://easybrain.com/uploads/media/1920x1080/08/108-New%20Easybrain%20puzzle%20BlockuDoku%20is%20now%20available%20on%20the%20App%20Store%20worldwide.%20.jpg?v=1-0)

### The original game
* At each point 3 blocks are drawn (not at random, somehow based on the current state).
* The player chooses in what order they place the blocks. If all 3 are placed, another batch is served.
* Points granted after every block placement and every time the board is cleared + additional move combos rewards.

### Blockulib version
* At each turn, 1 block is drawn (uniformly at random).
* If the player cannot place the block anywhere on the board, the game ends.  
* One point is granted for each turn the player survives.
