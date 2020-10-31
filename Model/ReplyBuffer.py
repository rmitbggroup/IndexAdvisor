import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, LEARNING_START):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.valid_len = 0
        self.LEARNING_START = LEARNING_START

    def can_update(self):
        return self.valid_len > self.LEARNING_START

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
            self.valid_len += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, self.valid_len, size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)