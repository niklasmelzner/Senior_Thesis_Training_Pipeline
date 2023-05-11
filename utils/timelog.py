"""
API used to log delays
Usage:
log = start()
...
log("Passed time: {0}")
"""
import time


class TimeLog:

    def __init__(self):
        self.t = time.time()

    def log(self, label: str):
        """{0} contained in label gets replaced with passed duration since last call or creation"""
        print(label.format(round(time.time() - self.t, 3)))
        self.t = time.time()

    def __call__(self, label: str):
        self.log(label)


def start() -> TimeLog:
    return TimeLog()
