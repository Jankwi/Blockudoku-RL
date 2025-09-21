import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from blockulib.utils import DeepList

def display_board(board):
    cmap = ListedColormap(["white", "blue"])

    plt.imshow(board, cmap=cmap, origin='upper', interpolation='nearest')
    plt.title("9x9 Binary Grid")
    plt.show()
    
def deep_display(PLAYING_LOOP, playing_config = {}):
    index = 3
    loop = PLAYING_LOOP(pos_list_type = DeepList)
    pos = loop(num_games = 10, **playing_config)
    
    print(f"Game Length = {len(pos[index])}")
    for board in pos[index]:
        display_board(board)