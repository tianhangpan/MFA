import time
from datetime import datetime


class TimeEstimator:
    def __init__(self):
        super().__init__()
        self.last_time = None
        self.now_time = None
        self.simple_time = None

    def mark(self):
        self.last_time = self.now_time
        self.now_time = time.time()

    def estimate(self, now, target):
        if self.last_time and self.now_time:
            interval = self.now_time - self.last_time
            expected_end_time = datetime.fromtimestamp(time.time() +
                                                       interval * (target - now))
            str_eet = datetime.strftime(expected_end_time, '%Y-%m-%d %H:%M:%S')
            print(f'the programme is expected to end at: {str_eet}')

    def simple_mark(self):
        self.simple_time = time.time()

    def query_time_span(self):
        return int(time.time() - self.simple_time)


if __name__ == '__main__':
    estimator = TimeEstimator()
    estimator.simple_mark()
    time.sleep(3)
    estimator.query_time_span()
