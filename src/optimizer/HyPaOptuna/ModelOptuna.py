import optuna
from evaluation.metrics import accuracy
from keras import backend as K

class ModelOptuna():
    """
    - normally you should adapt the values to your training curve
    - as long as the error keeps dropping, as well as the validation error, you should keep do training
    - better implement early stopping

    This was the better architecture, but optuna cant work with lambda stuff...

            hypas = [
                ('n_epochs', lambda name: trial.suggest_int(name=name, low=5, high=30)),
                ('batch_size', lambda name: trial.suggest_int(name=name, low=16, high=64)),
                ('learning_rate', lambda name: trial.suggest_loguniform(name=name, low=0.001, high=0.1))
            ]
            hypa_dict = {name: lambda: hypa_func(name) for name, hypa_func in hypas}

            # get and set new variables
            for name, hypa_generator in hypa_dict.items():
                self.current_model_genome.__setattr__(name, hypa_generator())
                
    """
    def _test(self, n_epochs, n_batch_size, lr):
        # set learning rate
        K.set_value(self.model.optimizer.learning_rate, lr)

        # save default weights
        self.model.save_weights('data/model.h5')
        # load default weights to reset model weights
        self.model.load_weights('data/model.h5')
        return 0.1

    def _model_fit_test(self, n_epochs, n_batch_size, lr):

        # set learning rate
        K.set_value(self.model.optimizer.learning_rate, lr)

        # save default weights
        self.model.save_weights('data/model.h5')
        self.model.fit(
            self.X_train_fit, 
            self.y_train_fit,
            epochs=n_epochs,
            batch_size=n_batch_size,
        )

        y_test_pred = self.model.predict(self.X_test_fit)

        # load default weights to reset model weights
        self.model.load_weights('data/model.h5')
        fitness = accuracy(self.y_test_fit, y_test_pred)
        return fitness

    def __init__(self, model, n_trials, X_train_fit, y_train_fit, X_test_fit, y_test_fit, log_func):
        self.testing = False
        self.model = model
        self.n_trials = n_trials
        self.X_train_fit, self.y_train_fit, self.X_test_fit, self.y_test_fit = X_train_fit, y_train_fit, X_test_fit, y_test_fit
        self.log_func = log_func

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