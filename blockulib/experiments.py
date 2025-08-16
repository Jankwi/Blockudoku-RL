import torch
from time import perf_counter
import matplotlib.pyplot as plt
from blockulib.utils import PositionList, ShallowList, DeepList

def test_playing_loop_timing(playing_loop, max_exp: int = 1, pos_list_type: PositionList = DeepList, show_plot=True, reps = 8):
    
    xs, ys = torch.logspace(0, max_exp, steps=max_exp+1, base=2, dtype=torch.int64).tolist(), []
    loop = playing_loop(pos_list_type = pos_list_type)
    
    for n in xs:
        start_time = perf_counter()
        for i in range(reps):
            loop(num_games=n)
        end_time = perf_counter()
        
        execution_time = end_time - start_time
        ratio = execution_time / (n * reps)
        ys.append(ratio)
        print(f"Playing {n} games at once {reps} times took {execution_time}: ratio of {ratio:.6e} s/game")
    
    if show_plot:
        plt.figure(); plt.plot(xs, ys, marker="o"); plt.xscale("log", base=2); plt.show()
        
    best = xs[ys.index(min(ys))]
    best_ratio = ys[ys.index(min(ys))]
    print(f"Best ratio achieved for {best} games at once with a ratio of {best_ratio:.6e} s/game")
    return best