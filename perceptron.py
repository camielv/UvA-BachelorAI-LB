# Perceptron
class Perceptron(object):
    def __init__(self, weights=None, threshold=0.5):
        self.threshold = threshold
        self.weights = weights
         
    def output(self, input_vector):
        if self._total(input_vector) < self.threshold:
            return 0
        else:
            return 1
 
    def train(self, training_set, alpha=0.1, end_after=100):
        # Initialise weights
        if self.weights is None:
            self.weights = [0 for _ in range(len(training_set.keys()[0]))]

        # Number of iterations
        n = 0

        updated = True

        # Training loop
        while updated:
            # Increment iterations
            n += 1
            updated = False
            
            # Iterate over all training elements
            for (xv, t) in training_set.items():
                y = self.output( xv )
                if y != t:
                    # If output doesn't match, update weights
                    self._update_weights(alpha, t, y, xv)
                    self._update_threshold(alpha, t, y)
                

            # Terminate after end_after iterations
            if end_after is not None and n >= end_after:
                break
            
        return n
    
    def set_weights(self,new_weights):
		# Set the weights to a given list
        self.weights = new_weights

    def reset(self):
        # Reset weights and threshold
        self.weights = None
        self.threshold = 0.5
 
    def test(self, training_set):
        for xv, t in training_set.items():
            if self.output(xv) != t:
                return False
        return True
 
    def _total(self, input_vector):
        total = 0
        for w, x in zip(self.weights, input_vector):
            total += (w * x)
        return total
 
    def _update_weights(self, alpha, t, y, xv):
        for i in range(len(self.weights)):
            self.weights[i] = (alpha * (t - y) * xv[i]) + self.weights[i]
 
    def _update_threshold(self, alpha, t, y):
        self.threshold = (alpha * (t - y) * -1) + self.threshold
