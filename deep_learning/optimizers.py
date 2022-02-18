class SGD():
    def __init__(self, lr) -> None:
        self.lr = lr

    def opt(self, weights, dw):
        # print(np.sum(np.abs(dw)))
        return weights - (self.lr * dw)
