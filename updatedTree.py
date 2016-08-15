import sys, random, math, copy
from itertools import chain, imap, product


class BinaryNode(object):
    def __init__(self, value=None, arity=2, children=[]):
        self.value = value
        self.children = children
        self.arity = arity

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def __iter__(self):
        for v in chain(*imap(iter, self.children)):
            yield v
        yield self.value

    def evaluateCircuit(self, combList):
        result = list()
        i = 0
        A = []
        # For each combination of 1s and 0s, evaluate:
        for combination in combList:
            for j in range(len(combination)):
                # Prepare the array A to pass the correct value of combination.
                # Example: (0, 0, 0)
                #         (0, 0, 1) ...
                A.append(int(combination[j]))
            # evaluateRec(A) is responsible for do the actual evaluation, with the A values passed that were crafted.
            result.append(int(self.evaluateRec(A)))

            A[:] = []
            i += 1
        return result

    def evaluateRec(self, A):
        # Recursively evaluates the result of the circuit.
        if self.value == 'and':
            result = 1
            for i in range(len(self.children)):
                result = result and self.children[i].evaluateRec(A)
            return result
        elif self.value == 'or':
            result = 0
            for i in range(len(self.children)):
                result = result or self.children[i].evaluateRec(A)
            return result
        elif self.value == 'not':
            return not self.children[0].evaluateRec(A)
        else:
            return eval(self.value)

    def writeFunc(self, array):
        # Writes the circuit. Inline mode.
        if len(self.children) > 1:
            array.append('(')
            for i in range(len(self.children) - 1):
                self.children[i].writeFunc(array)
                array.append(self.value)
            self.children[len(self.children) - 1].writeFunc(array)
            array.append(')')
        elif len(self.children) == 1:
            array.append('(')
            array.append(self.value)
            self.children[len(self.children) - 1].writeFunc(array)
            array.append(')')
        else:
            array.append(self.value)
        return array

    def addBinaryNode(self, binaryNode):
        # Adds a BinaryNode on the first possible location.
        if len(self.children) < self.arity:
            self.children.append(binaryNode)
            return True
        elif self.arity > 0:
            for child in self.children:
                if child.addBinaryNode(binaryNode):
                    return True
        else:
            # Couldn't add a BinaryNode anywhere in this tree.
            return False

    def getSize(self):
        # returns the size of the tree, doing the recursive way.
        results = []
        nodes = self.children
        while 1:
            newNodes = []
            if len(nodes) == 0:
                break
            for node in nodes:
                results.append(node.value)
                if len(node.children) > 0:
                    for child in node.children:
                        newNodes.append(child)
            nodes = newNodes
        return len(results)

    def writeFuncIt(self):
        results = []
        results.append(self.value)
        nodes = self.children
        while 1:
            newNodes = []
            if len(nodes) == 0:
                break
            for node in nodes:
                results.append(node.value)
                if len(node.children) > 0:
                    for child in node.children:
                        newNodes.append(child)

            nodes = newNodes
        return results

    def selectRandomNode(self):
        # Select a random Node and returns it. Iterative mode.
        size = self.getSize()
        counter = random.randint(1, size)
        nodes = self.children
        while 1:
            newNodes = []
            for node in nodes:
                counter -= 1
                if counter == 0:
                    return node
                if len(node.children) > 0:
                    for child in node.children:
                        newNodes.append(child)
            nodes = newNodes

    def mutation(self, listOperations, listTerminals):
        # Select random node to me mutated
        toBeMutated = self.selectRandomNode()
        # The node "not" can't be mutated, since it's the only one with arity = 1.
        tmpListOperations = listOperations[:]
        tmpListOperations.remove(['not', 1])

        # The mutation is done in a way to keep the tree valid. So operations are only replaced for operations
        # and terminals are only replaced for terminals
        if [toBeMutated.value, len(toBeMutated.children)] in tmpListOperations:
            random_Operation = random.randrange(0, len(tmpListOperations))
            toBeMutated.value = tmpListOperations[random_Operation][0]
        elif toBeMutated.value in listTerminals:
            random_Terminal = random.randrange(0, len(listTerminals))
            toBeMutated.value = listTerminals[random_Terminal]
            toBeMutated.children = []
        else:
            # "not" case
            return

    def crossOver(self, counter, t2):
        # Search for the node that is going to be replaced in the processs of cross-over.
        if counter == 0:
            return
        # If the desired node is a child of the present node, do:
        elif len(self.children) >= counter:
            tmp = t2.selectRandomNode()
            # Notice that this changes the actual self.children. So the previous form of this individual need to be
            # copied if you want to keep information about it before the crossover.
            self.children[counter - 1] = copy.deepcopy(tmp)
            counter = 0
        # If it's not, continue searching for the desired node
        else:
            counter -= len(self.children)
            for child in self.children:
                if child.getSize() >= counter:
                    return child.crossOver(counter, t2)
                else:
                    counter -= child.getSize()
        return

    def calcFitness(self, combList, S):
        fitness = 0
        evaluation = self.evaluateCircuit(combList)
        # check if the tree surpasses the maximum size allowed.
        if self.getSize() > TREE_MAX_SIZE:
            fitness = 1000000
            return fitness
        # If the evaluation is smaller than the combList, something went wrong.
        elif len(combList) != len(evaluation):
            fitness = 1000000
            return fitness
        # Adds 100 cost to each wrong evaluation result
        else:
            for i in range(len(S)):
                if S[i] != evaluation[i]:
                    fitness += 100
        return fitness


class Population(object):
    def __init__(self, array=[]):
        self.array = array

    def appendRandomTree(self, listOperations, listTerminals, combList, S):
        # Add root to the tree. It's always an operation.
        # This is necessary to start the "emptyTerminals" counter.
        random_Operation = random.randrange(0, len(listOperations))
        newNode = BinaryNode(listOperations[random_Operation][0], listOperations[random_Operation][1], [])
        emptyTerminals = listOperations[random_Operation][1]
        while True:
            decisionMaking = random.randint(1, 100)
            # If emptyTerminals equals zero, there're no more empty spaces to add nodes in the tree.
            if emptyTerminals == 0:
                break
            # The probability of adding a terminal needs to be bigger than the one to add a function.
            elif decisionMaking < 30:
                random_Operation = random.randrange(0, len(listOperations))
                # Add a new node to the tree, selected randomically from the operation list.
                if newNode.addBinaryNode(
                        BinaryNode(listOperations[random_Operation][0], listOperations[random_Operation][1], [])):
                    emptyTerminals += listOperations[random_Operation][1] - 1
            else:
                random_Terminal = random.randrange(0, len(listTerminals))
                # Add a new node to the tree, selected randomically from the terminal list.
                if newNode.addBinaryNode(BinaryNode(listTerminals[random_Terminal], 0, [])):
                    emptyTerminals -= 1
        # Calculate the fitness of the tree created.
        self.array.append({'fitness': newNode.calcFitness(combList, S), 'tree': newNode})

    def initPopulation(self, size, listOperations, listTerminals, combList, S):
        # Create new random trees until the population size is reached.
        for i in range(0, size):
            self.appendRandomTree(listOperations, listTerminals, combList, S)

    def sortArray(self):
        tmp = sorted(self.array, key=lambda ind: ind['fitness'])
        self.array = tmp[:]

    def sortSingleArray(self, array):
        tmp = sorted(array, key=lambda ind: ind['fitness'])
        return tmp

    def reproduction(self, cycles, lO, combList, S):
        count = 0
        difference = 1000000000.0
        crossCandidates = []

        #        while(difference > 0):
        while (count < cycles):
            del crossCandidates[:]
            # Select randomically a number of different individuals, and sort them.
            # The best two are going to reproduction.
            while len(crossCandidates) < 10:
                randomIndex = random.randrange(len(self.array))
                # if self.array[randomIndex] not in crossCandidates:
                crossCandidates.append(self.array[randomIndex])
            crossCandidates = self.sortSingleArray(crossCandidates)[:]

            # Make copies of the individuals that are going to be reproducted
            newdict = copy.deepcopy(crossCandidates[0])
            newdict2 = copy.deepcopy(crossCandidates[1])

            self.array.append(newdict)
            self.array.append(newdict2)

            # Select a random index to be performed the crossOver on the individual
            crossIndex = random.randint(1, crossCandidates[0]['tree'].getSize())
            # Cross Over
            crossCandidates[0]['tree'].crossOver(crossIndex, newdict2['tree'])
            # Mutation
            crossCandidates[0]['tree'].mutation(lO, lT)
            # Calculate new individual's fitness
            crossCandidates[0]['fitness'] = crossCandidates[0]['tree'].calcFitness(combList, S)

            # Select a random index to be performed the crossOver on the individual
            crossIndex = random.randint(1, crossCandidates[1]['tree'].getSize())
            # Cross Over
            crossCandidates[1]['tree'].crossOver(crossIndex, newdict['tree'])
            # Mutation
            crossCandidates[1]['tree'].mutation(lO, lT)
            # Calculate new individual's fitness
            crossCandidates[1]['fitness'] = crossCandidates[1]['tree'].calcFitness(combList, S)

            self.sortArray()
            self.array.pop()
            self.array.pop()

            # Check the best individual of the population
            difference = self.array[0]['fitness']
            bestSize = self.array[0]['tree'].getSize()
            worst = self.array[len(self.array) - 1]['fitness']
            worstSize = self.array[len(self.array) - 1]['tree'].getSize()

            # Start a new generation
            count += 1

            #            plt.figure(1)
            #            plt.subplot(111)
            #            plt.plot(count, difference, 'bo', count, bestSize, 'k')
            # Did we find the desired result?
            if difference == 0:
                #                plt.show()
                break

            print difference, bestSize, worst, worstSize, count
        return


def createOperationList(number_inputs):
    return [['and', number_inputs], ['or', number_inputs], ['not', 1]]


def createTerminalList(number_inputs):
    lT = []
    for i in range(number_inputs):
        lT.append('A[' + str(i) + ']')
    return lT


def createCombList(number_inputs):
    return ["".join(seq) for seq in product("01", repeat=number_inputs)]





# lO = [['and',3],['or',3],['not',1]]
# lT = ['A[0]', 'A[1]', 'A[2]']
TREE_MAX_SIZE = 450
N_INPUTS = 3
lO = createOperationList(N_INPUTS)
lT = createTerminalList(N_INPUTS)
combList = createCombList(N_INPUTS)

print lO, lT, combList

# combList = [[0, 0], [0, 1], [1, 0], [1, 1]]
# combList = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
S = [0, 1, 1, 0, 1, 0, 0, 1]
# S = [0, 1, 1, 0, 1, 0 ,0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
# S = [0, 1, 1, 0]
population = Population()
population.initPopulation(100, lO, lT, combList, S)
population.sortArray()
# print "POPULATION: ", population.array
# size = 0
# print "SELECTED: ", population.array[0]['tree'].selectNode(3)
print "GENERATIONS: ", population.reproduction(100000, lO, combList, S)
print "BEST INDIVIDUAL: \n", population.array[0]['tree'], "SIZE: ", population.array[0]['tree'].getSize(), "FITNESS: ", \
population.array[0]['tree'].calcFitness(combList, S)
# print "Evaluate Circuit: ", population.array[0]['tree'].evaluateCircuit(combList)
# print "POPULATION> ", population.array
# print list(iter(population.array[0]['tree']))
# print population.array[0]['tree'].evaluateList(x_array, lO)
# print "evaluateRec: ", population.array[0]['tree'].evaluateCircuit(combList)
# print "writeFunc: ", population.array[0]['tree'].writeFuncIt()
# print "writeFunc: ", ' '.join([str(x) for x in population.array[0]['tree'].writeFunc([])])
# print "eval: ", eval(' '.join([str(x) for x in population.array[0]['tree'].writeFunc([])]))
# print "eval original: ", np.sum(np.power(population.array[0]['tree'].evaluate(x)-func, 2))
# population.array[0]['tree'].mutation(lO, lT)
# print "SIZE: ", population.array[0]['tree'].getSize()
# print "MUTATED: ", population.array[0]
