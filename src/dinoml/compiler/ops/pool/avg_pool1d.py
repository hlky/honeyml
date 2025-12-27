from dinoml.compiler.ops.pool.pool1d import pool1d_base


# pylint: disable=C0103
class avg_pool1d(pool1d_base):
    def __init__(self, kernel_size, stride, pad) -> None:
        super().__init__(stride, pad, kernel_size, "avg")
        self._attrs["op"] = "avg_pool1d"
