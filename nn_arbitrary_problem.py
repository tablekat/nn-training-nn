
import random

# An arbitrary problem solvable by a neural network, along with sets of solutions
# This is essentially a randomly generated classification problem with up to polynomial variations on the input
# For example, a network with 3 inputs (x,y,z), the problem might be classification based on (x^2 + 3y + z/10 < 3)
class ArbitraryProblem:

    def __init__(self, num_inputs, max_power = 2):
        self.num_inputs = num_inputs
        self.max_power = max_power
        # for max_power = 2, and num_inputs = 3, then `x^2 + 0.78y + 0.5y^2 < 0.899` would be represented as:
        # self.discriminator = 0.899
        # [0, 1, 0.78, 0.5, 0, 0] * [x, x^2, y, y^2, z, z^2]

        self.discriminator = random.uniform(-0.5, 0.5)
        self.coeffs = [0] * (num_inputs * max_power)
        self.coeffs = list(map(lambda x: random.uniform(-1, 1), self.coeffs))
        self.lessThan = True if random.uniform(0, 1) < 0.5 else False

    def asArray(self):
        a = [self.discriminator, 1 if self.lessThan else 0]
        a.extend(self.coeffs)
        return a

    @staticmethod
    def fromArray(arr, num_inputs, max_power = 2):
        r = ArbitraryProblem(num_inputs, max_power)
        r.discriminator = arr[0]
        r.lessThan = arr[1] > 0.5
        self.coeffs = arr[2:]
        return r

    def testInput(self, inp):
        if len(inp) != self.num_inputs:
            raise ValueError('Number of inputs must match for testInput in ArbitraryProblem')

        acc = 0
        for i in range(len(self.coeffs)):
            inputI = int(i / self.max_power)
            currPower = (i % self.max_power) + 1

            acc += pow(inp[inputI], currPower) * self.coeffs[i]

        if self.lessThan:
            return acc < self.discriminator
        else:
            return acc > self.discriminator

    def getTrainingInput(self):
        return list(map(lambda x: random.uniform(-2, 2), [0] * self.num_inputs))

    def getTrainingSet(self, num = 256):
        inputses = []
        outputs = []
        for i in range(num):
            inputs = self.getTrainingInput()
            res = [1, 0] if self.testInput(inputs) else [0, 1]
            inputses.append(inputs)
            outputs.append(res)
        return (inputses, outputs)

    def __str__(self):
        eqstrs = []
        for i in range(self.num_inputs):
            inputLetter = chr(ord('a') + i)
            for p in range(self.max_power):
                thisCoeff = self.coeffs[i * self.max_power + p]
                letterStr = "%.2f%s" % (thisCoeff, inputLetter)
                if p == 0:
                    eqstrs.append(letterStr)
                else:
                    eqstrs.append(letterStr + "^" + str(p + 1))

        compStr = "<" if self.lessThan else ">"
        return "ArbitraryProblem[%s %s %s]" % (" + ".join(eqstrs), compStr, str(self.discriminator))
