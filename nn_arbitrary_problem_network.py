
# docker build -t literally-what . && docker run -it literally-what python nn_arbitrary_problem_network.py

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from nn_arbitrary_problem import ArbitraryProblem

def getArbitraryNetwork():
    problemo = ArbitraryProblem(2, max_power=1)

    model = Sequential()
    model.add(Dense(8, input_dim=2))
    model.add(Dense(2)) # True/False
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy']) # binary_crossentropy # mean_squared_error #sgd

    inputs, results = problemo.getTrainingSet(num=5000)

    model.fit(inputs, results, validation_split=0.33, nb_epoch=2, batch_size=1, verbose=True) #verbose=True)

    return (problemo, model)

def modelToArray(model):
    # https://keras.io/layers/about-keras-layers/
    arr = []
    for layer in model.layers:
        layerNums = layer.get_weights() # list of numpy arrays
        biases = layerNums[1] # these are all zeros???
        weights = layerNums[0]
        for x in weights:
            arr.extend(x)
    return arr

def arrayToModel(model, arr):
    arrI = 0
    for layer in model.layers:
        layerNums = layer.get_weights() # list of numpy arrays
        biases = layerNums[1] # these are all zeros???
        weights = layerNums[0]
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = arr[arrI]
                arrI += 1
        layer.set_weights([weights, biases])
    return model

# model = Sequential()
# model.add(Dense(8, input_dim=2))
# model.add(Dense(2)) # True/False
# aardvark = modelToArray(model)
# print "|||||||" + str(aardvark)
# print "|||||||" + str(modelToArray(arrayToModel(model, aardvark)))

if __name__ == "__main__":
    for prob in range(10):
        problemo = ArbitraryProblem(2, max_power=1)
        print problemo

        model = Sequential()
        model.add(Dense(8, input_dim=2))
        model.add(Dense(2)) # True/False
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy']) # binary_crossentropy # mean_squared_error #sgd

        inputs, results = problemo.getTrainingSet(num=5000)

        model.fit(inputs, results, validation_split=0.33, nb_epoch=2, batch_size=1, verbose=True) #verbose=True)

        for i in range(3):
            ninputs, nresults = problemo.getTrainingSet(256)
            # for i in range(len(ninputs)):
            #     print "Input: %s | True res: %s" % (str(ninputs[i]), str(nresults[i]))
            #     modelOut = model.predict(ninputs[i], verbose=0)
            #     print "Input: %s | True res: %s | Model res: %s" % (str(ninputs[i]), str(nresults[i]), str(modelOut))

            # for i in range(len(ninputs)):
            #     scores = model.evaluate(ninputs[i], nresults[i], verbose=0) #=0)
            #     print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

            scores = model.evaluate(ninputs, nresults, verbose=0) #=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
