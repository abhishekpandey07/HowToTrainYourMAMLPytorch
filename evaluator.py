class Evaluator(object):
    def __init__(self):
        self.predictions = []
        self.total_iterations = 0
        self.test_set_predictions = None

    def reset(self):
        self.predictions = []
        self.total_iterations = 0

    def callback(self,prediction,iteration_number):
        self.predictions.append(prediction)
        iteration_number += 1

    def process_predictions(self):
        raise NotImplementedError
        
