import torch
from blockulib.utils import PositionList, ShallowList, DeepList

class PositionListTests():
    
    def __call__(self, ):
        for shallow in [False, True]:
            self.init_test(False, shallow = shallow)
            self.init_test(True, shallow = shallow)
            self.process_chosen_moves_test(shallow = shallow)
        print("All tests passed")
    
    def init_test(self, init_from_games, shallow = False, num_games = 10):
        architecture = ShallowList if shallow else DeepList
        
        if init_from_games:
            games = torch.zeros((num_games, 9, 9))
            pos_list = architecture("whatever", games)
        else:
            pos_list = architecture(num_games = num_games)
        
        assert(num_games == pos_list.num_games)
        assert(num_games == pos_list.active_games)
        
        active_boards = pos_list.active_boards()
        assert(num_games == active_boards.shape[0])
        for i in range(num_games):
            assert(torch.equal(torch.zeros(9, 9), active_boards[i]))
        
        expected_call_val = None
        if shallow:
            expected_call_val = torch.zeros(num_games, 9, 9), torch.zeros(num_games)
        else:
            expected_call_val = [[torch.zeros(9, 9)] for i in range(num_games)]
        self.compare_call(expected_call_val, pos_list(), shallow = shallow)
        
    def compare_call(self, one, two, shallow = False):
        if shallow:
            return self.compare_call_shallow(one, two)
        else:
            return self.compare_call_deep(one, two)
        
    def compare_call_shallow(self, tensor1, tensor2):
        return torch.equal(tensor1[0], tensor2[0]) and torch.equal(tensor1[1], tensor2[1])
    
    def compare_call_deep(self, list1, list2):
        assert(len(list1) == len(list2))
        for i in range(len(list1)):
            assert(len(list1[i]) == len(list2[i]))
            for j in range(len(list1[i])):
                assert(torch.equal(list1[i][j], list2[i][j]))
                
                
    def process_chosen_moves_test(self, shallow = False):
        if shallow:
            return self.pchm_shallow_test()
        else:
            return self.pchm_deep_test()
        
    def pchm_shallow_test(self,):
        shallow_list = ShallowList(5)
        shallow_list.active_boards()

        tensor = torch.randn(5, 9, 9)
        tensor[0, 0, 0]  = float('nan')
        tensor[2, 4, 5]  = float('nan')

        expected_pos_tensor = tensor.clone()
        expected_pos_tensor[0] = torch.zeros(9, 9)
        expected_pos_tensor[2] = torch.zeros(9, 9)
        
        expected_game_lengths = torch.tensor([0., 1., 0., 1., 1.])
        expected_state = torch.tensor([False, True, False, True, True])
        
        shallow_list.process_chosen_moves(tensor)
        assert(torch.equal(expected_pos_tensor, shallow_list.pos_tensor))
        assert(torch.equal(expected_game_lengths, shallow_list.game_lengths))
        assert(torch.equal(expected_state, shallow_list.state))
        
    def pchm_deep_test(self):
        deep_list = DeepList(5)
        deep_list.active_boards()
        
        tensor = torch.randn(5, 9, 9)
        tensor[0, 0, 0]  = float('nan')
        tensor[2, 4, 5]  = float('nan')
        
        expected_pos_list = [[torch.zeros(9, 9)] for i in range(5)]
        for i in range(5):
            if i not in {0, 2}:
                expected_pos_list[i].append(tensor[i].clone())
        
        deep_list.process_chosen_moves(tensor)
        self.compare_call_deep(expected_pos_list, deep_list.pos_list)