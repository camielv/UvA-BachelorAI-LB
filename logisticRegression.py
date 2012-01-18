# Implements Logistic Regression
import math

class LogisticRegression(object):
    def __init__(self, theta = [0.5, 0.5], alpha = 0.05):
        self.theta = theta;
        self.alpha = alpha;

    def _h(self, theta, x):
        result = list()
        for i in range(len(theta)):
            result.append( theta[i] * x[i] )
        return 1 / float( 1 + math.exp( -sum(result) ) )

    def _costFunction(self, theta, samples):
        cost = -1/len(samples);
        temp = 0;
        for i in range(len(samples)):
            x = [ 1, samples[i][0] ]
            y = samples[i][1]
            temp += ( y * math.log( self._h( theta, x ) ) + ( 1 - y ) * math.log( 1 - self._h( theta, x ) ) )
        return cost * temp

    def _gradientDescent(self, theta, samples, a):
        newTheta = theta;
        for i in range(len(theta)):
            temp = 0
            for j in range(len(samples)):
                x = [ 1, samples[j][0] ]
                y = samples[j][1]
                temp += ( self._h( theta, x ) - y ) * x[i]
            newTheta[i] = theta[i] - a * temp
        return newTheta
    
    def train(self, samples, iterations=10000):
        for i in range(iterations):
            self.theta = self._gradientDescent(self.theta, samples, self.alpha)
            print "Cost: ", self._costFunction(self.theta, samples)
        return self.theta
