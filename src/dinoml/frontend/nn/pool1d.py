from dinoml.compiler.ops import avg_pool1d
from dinoml.frontend.nn.module import Module


class AvgPool1d(Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.op = avg_pool1d(kernel_size, stride, padding)

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        return self.op(x)
