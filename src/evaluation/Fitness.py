from evaluation.metrics import accuracy, f1_score_
from utils.progress_bar import print_progress_bar
import numpy as np
import utils.settings as settings
import json


class Fitness():
    def __init__(self, X_train, y_train, X_test, y_test, X_y_validation_splits, validation_iterations, verbose=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_y_validation_splits = X_y_validation_splits
        self.validation_iterations = validation_iterations
        self.log_func = print if verbose else lambda *args: None
    
    def _test_percentages(self, y_train_check, y_test_check):
        
        # Two count dicts
        train_dict = {}
        test_dict = {}
        for i, y_set in enumerate([y_train_check, y_test_check]):
            check_dict = train_dict if i == 0 else test_dict
            for label in y_set:
                activity_key = f"label_{label}"
                if activity_key not in check_dict:
                    check_dict[activity_key] = 1
                else:
                    check_dict[activity_key] += 1
        
        # Output dict
        keys = set(list(train_dict.keys()) + list(test_dict.keys()))
        output_dict = {}

        for key in keys:
            if key not in train_dict:
                output_dict[key] = 1
            elif key not in test_dict:
                output_dict[key] = 0
            else:
                output_dict[key] = test_dict[key] / (train_dict[key] + test_dict[key])

        output_dict = {k: v for k, v in sorted(output_dict.items(), key=lambda item: item[0], reverse=False)}
        return output_dict
    
    def log_split_insights(self):

        def log_percentage(title, y_train_c, y_test_c):
            self.log_func(f"{title} Test-Percentages")
            self.log_func(f"{json.dumps(self._test_percentages(y_train_c, y_test_c), indent=4)}\n")
        
        log_percentage("Big Split", self.y_train, self.y_test)

        # self.X_y_validation_splits: [(X_train_01, y_train_01, X_val_01, y_val_01), (X_train_02, y_train_02, X_val_02, y_val_02), ...]
        idx = 0
        for small_split in self.X_y_validation_splits:
            _, y_train_i, _, y_test_i = small_split
            log_percentage(f"Small Split {idx}", y_train_i, y_test_i)
            idx += 1
    
    def _model_fit_test(self, model_genome, X_train_fit, y_train_fit, X_test_fit, y_test_fit, eval_func):
        model = model_genome.get_model()
        model.fit(
            X_train_fit, 
            y_train_fit, 
            batch_size=model_genome.batch_size, 
            epochs=model_genome.n_epochs,
            verbose=1
        )
        model.summary()
        y_test_pred = model.predict(X_test_fit)
        fitness = eval_func(y_test_fit, y_test_pred)
        return fitness
    
    def _kfold_fit_test(self, model_genome, X_y_val_splits, eval_func):
        prog_bar = lambda progress: print_progress_bar(progress, total=len(self.X_y_validation_splits), prefix="k_fold", suffix=f"{progress}/{len(X_y_val_splits)}", length=30, log_func=self.log_func, fill=">")

        fitnesses = []
        idx = 0
        for X_train_split, y_train_split, X_val_split, y_val_split in X_y_val_splits:
            prog_bar(progress=idx)
            idx += 1
            fitness = self._model_fit_test(model_genome=model_genome, 
                X_train_fit=X_train_split, 
                y_train_fit=y_train_split, 
                X_test_fit=X_val_split, 
                y_test_fit=y_val_split, 
                eval_func=eval_func
            )
            self.log_func(f"kfold {idx} fitness: {fitness}")
            fitnesses.append(fitness)
        return np.mean(fitnesses)

    
    def small_split_kfold_acc(self, model_genome, log_func) -> float:
        return self._kfold_fit_test(
            model_genome=model_genome, 
            X_y_val_splits=self.X_y_validation_splits, 
            eval_func=accuracy
        )
    
    def small_split_kfold_f1(self, model_genome, log_func) -> float:
        return self._kfold_fit_test(
            model_genome=model_genome, 
            X_y_val_splits=self.X_y_validation_splits, 
            eval_func=f1_score_
        )

    def big_split_acc(self, model_genome, log_func) -> float:
        return self._model_fit_test(model_genome=model_genome, 
            X_train_fit=self.X_train, 
            y_train_fit=self.y_train, 
            X_test_fit=self.X_test, 
            y_test_fit=self.y_test, 
            eval_func=accuracy
        )
    
    def small_split_kfold_max_val_iter_f1(self, model_genome, log_func) -> float:
        return self._kfold_fit_test(
            model_genome=model_genome, 
            X_y_val_splits=self.X_y_validation_splits[:self.validation_iterations], 
            eval_func=f1_score_
        )
    
    def small_split_kfold_max_val_iter_acc(self, model_genome, log_func) -> float:
        return self._kfold_fit_test(
            model_genome=model_genome, 
            X_y_val_splits=self.X_y_validation_splits[:self.validation_iterations], 
            eval_func=accuracy
        )