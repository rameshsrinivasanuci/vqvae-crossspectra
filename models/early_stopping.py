import numpy as np
import torch


# used to end training early if validation doesnt improve after a given patience.
class EarlyStopping:
    def __init__(self, patience = 20, delta = 0, path="../checkpoint.pt"):
        # amount of epochs in a row in which validation loss doesnt improve.
        self.patience = patience
        # the threshold to qualify as improvement.
        self.delta = delta
        self.path = path

        # holder of best validation loss
        self.best_score = None
        # flag for early stop
        self.early_stop = False
        
        self.counter = 0

        self.val_loss_min = np.inf

    # makes an instance of a class behave like a function.
    def __call__(self, val_loss, model):

        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # if the loss is worse than the best score.
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True

        # if it's not worse.
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

