from evaluation.metrics import accuracy
from utils.progress_bar import print_progress_bar
import numpy as np
import utils.settings as settings


class Fitness():
    def __init__(self, X_train, y_train, X_test, y_test, X_y_validation_splits, validation_iterations):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_y_validation_splits = X_y_validation_splits
        self.validation_iterations = validation_iterations

    def kfold_without_test_set(self, model_genome, log_func) -> float:
        prog_bar = lambda progress: print_progress_bar(progress, total=len(self.X_y_validation_splits[:self.validation_iterations]), prefix="k_fold", suffix=f"{progress}/{len(self.X_y_validation_splits[:self.validation_iterations])}", length=30, log_func=log_func, fill=">")
        # Refactoring idea
        # model_genome.fit(X_train, y_train)

        accuracies = []
        idx = 0
        for X_train_split, y_train_split, X_val_split, y_val_split in self.X_y_validation_splits[:self.validation_iterations]:
            prog_bar(progress=idx)
            idx += 1
            model = model_genome.get_model()
            model.fit(
                X_train_split, 
                y_train_split, 
                batch_size=model_genome.batch_size, 
                epochs=model_genome.n_epochs,
                verbose=0
            )
            y_val_pred = model.predict(X_val_split)
            accuracies.append(accuracy(y_val_split, y_val_pred))

        prog_bar(progress=len(self.X_y_validation_splits[:self.validation_iterations]))

        fitness = np.mean(accuracies)
        return fitness

    def normal_with_test_set(self, model_genome, log_func) -> float:
        model = model_genome.get_model()
        model.fit(
            self.X_train, 
            self.y_train, 
            batch_size=32, 
            epochs=1,
            verbose=1
        )
        model.summary()

        # model_genome.batch_size
        # model_genome.n_epochs

        y_test_pred = model.predict(self.X_test)
        fitness = accuracy(self.y_test, y_test_pred)
        return fitness