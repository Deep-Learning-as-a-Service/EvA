from evaluation.metrics import accuracy
import time
class Tester():
    def __init__(self, path, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.path = path
    
    def log_test_accuracy(self, model_genome, current_gen_idx, time):
        model = model_genome.get_model()

        model.fit(
            self.X_train, 
            self.y_train, 
            batch_size=model_genome.batch_size, 
            epochs=model_genome.n_epochs,
            verbose=0
        )
        y_test_pred = model.predict(self.X_test)
        with open(self.path, "a+") as f:
            f.write(f"{current_gen_idx} : {time} : {accuracy(y_test_pred, self.y_test)} \n")
