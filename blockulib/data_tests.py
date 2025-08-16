from blockulib.data import get_unique, MostPopular, NMostPopular, YDiscounter
import torch

class DataOrganizerTests():
    def __call__(self):
        self.get_unique_test()
        self.most_popular_test()
        NMostPopularTests()()
        print("All tests passed")
        
    def get_unique_test(self,):
        example_x = torch.tensor([[[1., 2.], [1., 2.]], [[0., 2.], [1., 2.]], [[0., 2.], [0., 2.]], [[1., 2.], [1., 2.]]])
        unique_blocks, inverse, counts = get_unique(example_x)
    
        expected_unique_blocks = torch.tensor([[[0., 2.],[0., 2.]], [[0., 2.],[1., 2.]], [[1., 2.],[1., 2.]]])
        expected_inverse = torch.tensor([2, 1, 0, 2])
        expected_counts = torch.tensor([1, 1, 2])
    
        assert(torch.equal(expected_unique_blocks, unique_blocks))
        assert(torch.equal(expected_inverse, inverse))
        assert(torch.equal(expected_counts, counts))
        
    def most_popular_test(self,):
        example_x_list = [torch.tensor([[[0., 0.], [0., 0.]], [[2., 1.], [2., 1.]]]), torch.tensor([[[0., 0.], [0., 0.]]]), torch.tensor([[[2., 1.], [2., 1.]], [[1., 0.], [0., 1.]]])]
        example_y_list = [torch.tensor([5., -1.]), torch.tensor([2.]), torch.tensor([1., 1.])]
        x, y = MostPopular().choose_boards(example_x_list, example_y_list, threshold = 2)

        expected_x = torch.tensor([[[0., 0.], [0., 0.]], [[2., 1.], [2., 1.]]])
        expected_y = torch.tensor([3.5, 0.])

        assert(torch.equal(expected_x, x))
        assert(torch.equal(expected_y, y))
        
class YTransformTests():
    def __call__(self):
        self.discounter_test()
        print("All tests passed")

    def discounter_test(self,):
        lengths = torch.tensor([1, 2, 2, 3, 1, 0, 4])
        disc = YDiscounter(dsc_rate = 0.5)

        vals = disc.lengths_to_values(lengths)
        expected_vals = torch.tensor([1.0000, 1.5000, 1.5000, 1.7500, 1.0000, 0.0000, 1.8750])
        assert(torch.equal(expected_vals, vals))

# LLM-written (GPT 5 - thinking) unit tests, reviewed by a human
class NMostPopularTests():
    def __call__(self):
        self.no_tie_test()
        self.tie_break_test()

    def _as_tuples(self, blocks):
        # Flatten each (2x2) block to a tuple so we can compare sets ignoring order
        return {tuple(block.reshape(-1).tolist()) for block in blocks}

    def no_tie_test(self):
        # Blocks:
        A = torch.tensor([[0., 0.],[0., 0.]])  # appears 3x
        B = torch.tensor([[2., 1.],[2., 1.]])  # appears 2x
        C = torch.tensor([[1., 0.],[0., 1.]])  # appears 1x

        x_list = [
            torch.stack([A, B]),     # A, B
            torch.stack([A]),        # A
            torch.stack([B, C])      # B, C
        ]
        y_list = [torch.zeros(len(x)) for x in x_list]  # y not used

        top_n = 2
        out = NMostPopular().transform(x_list, y_list, top_n)

        # Expect {A, B} (no tie needed)
        expected = torch.stack([A, B])
        assert out.shape[0] == top_n
        assert self._as_tuples(out) == self._as_tuples(expected)

    def tie_break_test(self):
        # Make a tie for the final slot:
        # A=3, B=2, C=2, D=1 ; top_n=2  -> must contain A and exactly one of {B, C}
        A = torch.tensor([[0., 0.],[0., 0.]])
        B = torch.tensor([[2., 1.],[2., 1.]])
        C = torch.tensor([[1., 0.],[0., 1.]])
        D = torch.tensor([[9., 9.],[9., 9.]])

        x_list = [
            torch.stack([A, B]),   # A, B
            torch.stack([A, C, B]),   # A, C
            torch.stack([A]),      # A
            torch.stack([D]),      # D
            torch.stack([C])       # C (now B=2, C=2)
        ]
        y_list = [torch.zeros(len(x)) for x in x_list]

        top_n = 2

        out = NMostPopular().transform(x_list, y_list, top_n)
        assert out.shape[0] == top_n

        out_set = self._as_tuples(out)
        A_t = tuple(A.reshape(-1).tolist())
        B_t = tuple(B.reshape(-1).tolist())
        C_t = tuple(C.reshape(-1).tolist())

        # Must include A
        assert A_t in out_set
        # The other must be either B or C (but not both, since top_n=2)
        assert (B_t in out_set) ^ (C_t in out_set)

        