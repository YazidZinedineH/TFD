# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test model button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Getting Started Question
#
# Given this data, train a neural network to match the xs to the ys
# So that a predictor for a new value of X will give a float value
# very close to the desired answer
# i.e. print(model.predict([10.0])) would give a satisfactory result
# The test infrastructure expects a trained model that accepts
# an input shape of [1]

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    # YOUR CODE HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('loss') < 1e-4):
                print('\n MSE loss is lesser than 1e-4')
                print('\n Stop training')
                self.model.stop_training = True
    callbak = myCallback()
    model = keras.Sequential([keras.layers.Dense(units=32, input_shape=[1]),
                              keras.layers.Dense(units=1)])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=100, callbacks=[callbak])
    print(model.predict([10.0]))
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
