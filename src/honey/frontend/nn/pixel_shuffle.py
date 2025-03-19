from honey.compiler.ops import pixel_shuffle, pixel_unshuffle
from honey.frontend.nn.module import Module


class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.op = pixel_shuffle(upscale_factor)

    def forward(self, *args):
        x = args[0]
        return self.op(x)


class PixelUnshuffle(Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.op = pixel_unshuffle(downscale_factor)

    def forward(self, *args):
        x = args[0]
        return self.op(x)
