import optuna

class HyPaOptuna():
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
    def __init__(self, input_model_genome, n_trials, fitness_func, log_func):
        self.current_model_genome = input_model_genome
        self.n_trials = n_trials
        self.log_func = log_func

        self.get_fitness = lambda: fitness_func(model_genome=self.current_model_genome, log_func=self.log_func)

    def run(self):

        default_fitness = self.get_fitness()

        def objective(trial):

            self.current_model_genome.n_epochs = trial.suggest_int('n_epochs', low=5, high=30)
            self.current_model_genome.batch_size = trial.suggest_int('batch_size', low=16, high=64)
            self.current_model_genome.learning_rate = trial.suggest_loguniform('learning_rate', low=0.0001, high=0.002)

            return 1 - self.get_fitness()

        # Do optimization
        study = optuna.create_study()  # Create a new study.
        study.optimize(objective, n_trials=self.n_trials)  # Invoke optimization of the objective function.

        # Test Optuna model on normal fitness
        for name, value in study.best_params.items():
            self.current_model_genome.__setattr__(name, value)
        optuna_fitness = self.get_fitness()

        self.log_func(f'*** HypaFinished')
        self.log_func(f'Default fitness: {default_fitness}')
        self.log_func(f'HypaOptuna fitness: {optuna_fitness}')
        self.log_func(f'HypaOptuna best params: {study.best_params}')
        
        if optuna_fitness < default_fitness:
            self.current_model_genome.reset_to_default_params()
            
        return self.current_model_genome