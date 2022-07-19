import optuna
from evaluation.metrics import accuracy
from keras import backend as K
import numpy as np

class ModelOptuna2():

    def _test(self, n_epochs, n_batch_size, lr):
        keras_model = None

        if self.is_seqevo:
            keras_model = self.model.get_model()
        else:
            keras_model = self.model._create_model()
        K.set_value(keras_model.optimizer.learning_rate, lr)
        keras_model.fit(
            self.X_train_fit, 
            self.y_train_fit,
            epochs=1,
            batch_size=n_batch_size,
        )

        y_test_pred = keras_model.predict(self.X_test_fit)

        # load default weights to reset model weights
        fitness = accuracy(self.y_test_fit, y_test_pred)
        return fitness

    def _model_fit_test(self, n_epochs, n_batch_size, lr):
        
        fitnesses = []
        idx = 0
        for X_train_split, y_train_split, X_val_split, y_val_split in self.X_y_val_splits: 
            keras_model = None

            if self.is_seqevo:
                keras_model = self.model.get_model()
            else:
                keras_model = self.model._create_model()

            # set learning rate
            K.set_value(keras_model.optimizer.learning_rate, lr) 
            idx += 1
            
            keras_model.fit(
                X_train_split, 
                y_train_split,
                epochs=n_epochs,
                batch_size=n_batch_size,
            )

            y_test_pred = keras_model.predict(X_val_split)

            # load default weights to reset model weights
            fitness = accuracy(y_val_split, y_test_pred)
            fitnesses.append(fitness)
        
        return np.mean(fitnesses)

    def __init__(self, model, n_trials, X_y_val_splits, log_func, is_seqevo):
        self.testing = False
        self.model = model
        self.n_trials = n_trials
        self.X_y_val_splits = X_y_val_splits
        self.log_func = log_func
        self.is_seqevo = is_seqevo
        self.best_fitness = 0

        self.get_fitness = self._model_fit_test if not self.testing else self._test

    def run(self):
        def objective(trial):

            n_epochs = trial.suggest_int('n_epochs', low=5, high=30)
            batch_size = trial.suggest_int('batch_size', low=16, high=64)
            learning_rate = trial.suggest_loguniform('learning_rate', low=0.0001, high=0.002)
            fitness = self.get_fitness(n_epochs, batch_size, learning_rate)
            self.best_fitness = max(self.best_fitness, fitness)
            self.log_func(f"this trial: {fitness}, global best: {self.best_fitness}")
            return 1 - fitness

        # Do optimization
        study = optuna.create_study()  # Create a new study.
        study.optimize(objective, n_trials=self.n_trials)  # Invoke optimization of the objective function.

            
        return study.best_params