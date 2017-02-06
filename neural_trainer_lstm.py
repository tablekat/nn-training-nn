
# docker build -t literally-what . && docker run -it literally-what python neural_trainer_lstm.py

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from nn_arbitrary_problem import ArbitraryProblem
from nn_arbitrary_problem_network import getArbitraryNetwork, modelToArray, arrayToModel

arbitraryProblems = []
arbitraryModels = []
arbitraryModelArrs = []
for i in range(64):
    problem, model = getArbitraryNetwork()
    arbitraryProblems.append(problem.asArray())
    arbitraryModels.append(model)
    arbitraryModelArrs.append(modelToArray(model))
