import types
import time


def to_string(**kwargs):
    return ' '.join([f"{k}:{v}" for k, v in kwargs.items()])


class Timer:
    def __init__(self, debugText="", print_funct: types.FunctionType = print):
        self.debugText = debugText
        self.print_funct = print_funct

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        message = f"{self.debugText} Diff={self.interval}"
        self.print_funct(message)