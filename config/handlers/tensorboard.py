import logging
import time
from torch.utils.tensorboard import SummaryWriter


class BatchedTensorBoardHandler(logging.Handler):
    """
    Collects metric dicts, averages them, and writes to TensorBoard
    in batches to minimize overhead.
    """

    def __init__(self, writer: SummaryWriter, batch_size: int = 10, flush_secs: float = 5.0):
        super().__init__()
        self.writer = writer
        self.batch_size = batch_size
        self.flush_secs = flush_secs

        self.buffer = []
        self.last_flush = time.time()
        self.step = 0

    def emit(self, record):
        if not isinstance(record.msg, dict):
            return

        self.buffer.append(record.msg)
        self.step += 1

        time_due = (time.time() - self.last_flush) >= self.flush_secs
        size_due = len(self.buffer) >= self.batch_size

        if time_due or size_due:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        merged = {}
        for entry in self.buffer:
            for k, v in entry.items():
                merged.setdefault(k, []).append(v)

        for k, v in merged.items():
            self.writer.add_scalar(k, sum(v) / len(v), self.step)

        self.buffer.clear()
        self.last_flush = time.time()
