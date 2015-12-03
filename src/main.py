'''
Programming a Perceptron in Python

A perceptron classifier is a simple model of a neuron
'''

#!/usr/bin/env python

from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt


def unit_step(value):
    ''' Step function : represents the unit step function,
        equal to 0 for an value < 0 else 1.
    '''
    if value < 0:
        return 0
    else:
        return 1

# OR gate
training_data = [(array([0, 0, 1]), 0),
                 (array([0, 1, 1]), 1),
                 (array([1, 0, 1]), 1),
                 (array([1, 1, 1]), 1), ]

weight = random.rand(3)

errors = []
learning_rate = 0.2
iteration = 100

for i in xrange(iteration):
    x, expected = choice(training_data)
    result = dot(weight, x)
    error = expected - unit_step(result)
    errors.append(error)
    weight += learning_rate * error * x

for x, _ in training_data:
    result = dot(x, weight)
    print "{}: {} -> {}".format(x[:2], result, unit_step(result))

# Plot error graph
plt.plot(errors)
plt.show()
