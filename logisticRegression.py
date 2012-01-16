# Implements Logistic Regression
import math


class LogisticRegression(object):
    def __init__(self, theta = [0.5, 0.5], alpha = 0.05):
        self.theta = theta;
        self.alpha = alpha;

    def _h(self, theta, x):
        result = list()
        for i in range(len(theta)):
            result.append(theta(i) * x(i))
        return 1 / float( 1 + math.exp( -sum(result) ) )

    def _costFunction(self, theta, x, y):
        cost = -1/len(x);
        temp = 0;
        for i in range(len(x)):
            temp += ( y[i] * math.log( h( theta, x[i] ) ) + ( 1 - y[i] ) * math.log( 1 - h( theta, x[i] ) ) )
        return cost * temp

    def _gradientDescent(self, theta, samples, a):
        newTheta = theta;
        for i in range(len(theta)):
            temp = 0
            for j in range(len(samples)):
                
                temp += ( h( theta, (samples[j][0] ) - y[j] ) * x[j][i]
             newTheta[i] = theta[i] - a * temp
        return newTheta
    
    def train(self, samples):
        for i in range(1000):
            self.theta = _gradientDescent(self.theta, samples, self.alpha)
        return self.theta
