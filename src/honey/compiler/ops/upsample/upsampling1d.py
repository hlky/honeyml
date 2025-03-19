from honey.compiler.ops.upsample.upsampling1d_base import upsampling1d_base


# pylint: disable=C0103
class upsampling1d(upsampling1d_base):
    def __init__(self, scale_factor, mode, align_corners=False) -> None:
        super().__init__(scale_factor, mode, align_corners)
        self._attrs["op"] = "upsampling1d"
        self._attrs["mode"] = mode
