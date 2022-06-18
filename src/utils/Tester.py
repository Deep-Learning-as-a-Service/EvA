from evaluation.metrics import accuracy
class Tester():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def get_test_accuracy(self, model_genome):
        model = model_genome.get_model()

        model.fit(
            self.X_train, 
            self.y_train, 
            batch_size=model_genome.batch_size, 
            epochs=model_genome.n_epochs,
            verbose=0
        )
        y_test_pred = model.predict(self.X_test)
        return accuracy(y_test_pred, self.y_test)
