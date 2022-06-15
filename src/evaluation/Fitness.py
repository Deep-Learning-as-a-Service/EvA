from evaluation.metrics import accuracy

class Fitness():
    def __init__(self, X_train, y_train, X_test, y_test, X_y_validation_splits, window_size, n_features, n_classes):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_y_validation_splits = X_y_validation_splits
        self.window_size = window_size
        self.n_features = n_features
        self.n_classes = n_classes
    
    def kfold_without_test_set(self, model_genome, log_func) -> float:
        prog_bar = lambda progress: print_progress_bar(progress, total=len(self.X_y_validation_splits), prefix="k_fold", suffix=f"{progress}/{len(self.X_y_validation_splits)}", length=30, log_func=log_func, fill=">")
        # Refactoring idea
        # model_genome.fit(X_train, y_train)

        # Traininsparams
        batch_size = model_genome.batch_size
        learning_rate = model_genome.learning_rate
        n_epochs = model_genome.n_epochs

        accuracies = []
        for idx, (X_train, y_train, X_val, y_val) in enumerate(self.X_y_validation_splits):
            prog_bar(progress=idx)
            model = model_genome.get_model(
                window_size=self.window_size,
                n_features=self.n_features,
                n_classes=self.n_classes
            )
            model.fit(
                self.X_train, 
                self.y_train, 
                batch_size=model_genome.batch_size, 
                epochs=model_genome.n_epochs,
                verbose=0
            )
            y_val_pred = model.predict(X_val)
            accuracies.append(accuracy(y_val, y_val_pred))

        prog_bar(progress=len(self.X_y_validation_splits))

        fitness = np.mean(accuracies)
        return fitness

    def normal_with_test_set(self, model_genome, log_func) -> float:
        model = model_genome.get_model(
            window_size=self.window_size,
            n_features=self.n_features,
            n_classes=self.n_classes
        )
        model.fit(
            self.X_train, 
            self.y_train, 
            batch_size=32, 
            epochs=1,
            verbose=1
        )

        # model_genome.batch_size
        # model_genome.n_epochs

        y_test_pred = model.predict(self.X_test)
        fitness = accuracy(self.y_test, y_test_pred)
        return fitness
    
