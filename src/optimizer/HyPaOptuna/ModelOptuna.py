import optuna
from evaluation.metrics import accuracy
from keras import backend as K

class ModelOptuna():

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
        return 0.1

    def _model_fit_test(self, n_epochs, n_batch_size, lr):
        keras_model = None

        if self.is_seqevo:
            keras_model = self.model.get_model()
        else:
            keras_model = self.model._create_model()

        # set learning rate
        K.set_value(keras_model.optimizer.learning_rate, lr)
        keras_model.fit(
            self.X_train_fit, 
            self.y_train_fit,
            epochs=n_epochs,
            batch_size=n_batch_size,
        )

        y_test_pred = keras_model.predict(self.X_test_fit)

        # load default weights to reset model weights
        fitness = accuracy(self.y_test_fit, y_test_pred)
        return fitness

    def __init__(self, model, n_trials, X_train_fit, y_train_fit, X_test_fit, y_test_fit, log_func, is_seqevo):
        self.testing = False
        self.model = model
        self.n_trials = n_trials
        self.X_train_fit, self.y_train_fit, self.X_test_fit, self.y_test_fit = X_train_fit, y_train_fit, X_test_fit, y_test_fit
        self.log_func = log_func
        self.is_seqevo = is_seqevo

        self.get_fitness = self._model_fit_test if not self.testing else self._test

    def run(self):
        def objective(trial):

            n_epochs = trial.suggest_int('n_epochs', low=5, high=30)
            batch_size = trial.suggest_int('batch_size', low=16, high=64)
            learning_rate = trial.suggest_loguniform('learning_rate', low=0.0001, high=0.002)

            return 1 - self.get_fitness(n_epochs, batch_size, learning_rate)

        # Do optimization
        study = optuna.create_study()  # Create a new study.
        study.optimize(objective, n_trials=self.n_trials)  # Invoke optimization of the objective function.

            
        return study.best_params