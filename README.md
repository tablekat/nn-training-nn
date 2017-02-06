# nn-training-nn
Neural network which trains neural networks

This creates arbitrary decision problems randomly, then creates and trains neural networks with adadelta to solve these problems. They get to an accuracy of about 98%.

Then, the primary network will take desired inputs and outputs (preferablly way smaller training set than the adadelta training requires), and it will learn to output the proper weights for a correctly tuned neural network solution to the arbitrary problem. This will use a LSTM network, and will be fed input and output pairs in sequence, and be trained on a data set of generated and adadelta-trained networks. This will essentially create general neural network that will design new networks to solve decision problems of this class.

This is written with Keras and uses Docker~! Exciting.

## The Arbitrary Problems
Arbitrary problems are polynomials of n variables, with a disciminator to classify the result as true or false. The ones used here will be first-order polynomials, because that allows the simplest NN solutions.
Some example problems with 2 inputs and first order is:

    ArbitraryProblem[-0.18a + -0.63b < 0.342905823789]
    ArbitraryProblem[-0.47a + -0.21b > 0.167684492042]
    
 The inputs are randomly chosen in a [-2..2, -2..2] uniform square, and a training set is built by plugging these inputs into the inequality. Then an adadelta network is trained with 2 inputs, 8 hidden neurons in a single layer, and a softmax true/false output.
 Many of these networks are created and trained with different ArbitraryProblems as the base decision problem. Some training data, and the weights of the trained networks, are then used as input to the primary network
 
 ## The Primary Network
 This is the network that is trained to take inputs and outputs of the child networks, and then output weights for the child networks to use. This part is still in progress. Adventure!
