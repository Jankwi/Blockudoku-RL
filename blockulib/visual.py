import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from blockulib.utils import DeepList

def display_board(board):
    cmap = ListedColormap(["white", "blue"])

    plt.imshow(board, cmap=cmap, origin='upper', interpolation='nearest')
    plt.title("9x9 Binary Grid")
    plt.show()
    
def deep_display(PLAYING_LOOP):
    index = 3
    loop = PLAYING_LOOP(pos_list_type = DeepList)
    pos = loop(num_games = 10)
    
    print(f"Game Length = {len(pos[index])}")
    for board in pos[index]:
        display_board(board)