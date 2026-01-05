from dinoml.compiler.ops.pool.pool1d import pool1d_base


class avg_pool1d_compress_time(pool1d_base):
    def __init__(self) -> None:
        super().__init__(stride=2, pad=0, kernel_size=2, reduce_func="avg")
        self._attrs["op"] = "avg_pool1d_compress_time"
