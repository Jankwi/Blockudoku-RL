from blockulib.data import UniqueBased, MostPopular
import torch

class DataOrganizerTests():
    def __call__(self):
        self.get_unique_test()
        self.most_popular_test()
        
    def get_unique_test(self,):
        example_x = torch.tensor([[[1., 2.], [1., 2.]], [[0., 2.], [1., 2.]], [[0., 2.], [0., 2.]], [[1., 2.], [1., 2.]]])
        unique_blocks, inverse, counts = UniqueBased().get_unique(example_x, k = 2)
    
        expected_unique_blocks = torch.tensor([[[0., 2.],[0., 2.]], [[0., 2.],[1., 2.]], [[1., 2.],[1., 2.]]])
        expected_inverse = torch.tensor([2, 1, 0, 2])
        expected_counts = torch.tensor([1, 1, 2])
    
        assert(torch.equal(expected_unique_blocks, unique_blocks))
        assert(torch.equal(expected_inverse, inverse))
        assert(torch.equal(expected_counts, counts))
        
    def most_popular_test(self,):
        example_x_list = [torch.tensor([[[0., 0.], [0., 0.]], [[2., 1.], [2., 1.]]]), torch.tensor([[[0., 0.], [0., 0.]]]), torch.tensor([[[2., 1.], [2., 1.]], [[1., 0.], [0., 1.]]])]
        example_y_list = [torch.tensor([5., -1.]), torch.tensor([2.]), torch.tensor([1., 1.])]
        x, y = MostPopular().choose_boards(example_x_list, example_y_list, threshold = 2, k = 2)

        expected_x = torch.tensor([[[0., 0.], [0., 0.]], [[2., 1.], [2., 1.]]])
        expected_y = torch.tensor([3.5, 0.])

        assert(torch.equal(expected_x, x))
        assert(torch.equal(expected_y, y))
        