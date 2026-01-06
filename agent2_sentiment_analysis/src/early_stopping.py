class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        """
        patience: số epoch chấp nhận không cải thiện
        min_delta: mức cải thiện tối thiểu được xem là có ý nghĩa
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        """
        Trả về True nếu nên dừng training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
