def readable_allocated_memory(memory_bytes):
    """Convert output of torch.cuda.memory_allocated()
    to human readable format
    """
    memory_kilobytes = memory_bytes / 1024
    memory_megabytes = memory_kilobytes / 1024
    memory_gigabytes = memory_megabytes / 1024
    if memory_gigabytes > 1:
        return f"{memory_gigabytes:.2f} GB"
    elif memory_megabytes > 1:
        return f"{memory_megabytes:.2f} MB"
    elif memory_kilobytes > 1:
        return f"{memory_kilobytes:.2f} KB"
    else:
        return f"{memory_bytes:.2f} B"

        
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = 10e10
        self.early_stop = False

    def __call__(self, val_loss):
        relative_loss = (self.best_loss - val_loss) / self.best_loss * 100
        if relative_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif relative_loss < self.min_delta:
            self.counter += 1
            print(
                f"Early stopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                print("Early stopping")
                self.early_stop = True