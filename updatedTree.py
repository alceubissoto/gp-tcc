import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random, copy
from itertools import chain, imap, product
import Tkinter as tk
import logging
import datetime
import time

LARGE_FONT = ("Verdana", 12)
SMALL_FONT = ("Verdana", 8)
POPULATION_SIZE = 400
MUTATION_PROBABILITY = 0.05
TREE_MAX_SIZE = 63
N_INPUTS = 4
TOURNAMENT_SIZE = 7
CROSS_PROB = 0.5
NUMBER_OF_GENERATIONS = 1000000
lO = None
lT = None
xListBest = []
yListBestFitness = []
xListAverage = []
yListAverageFitness = []
f = Figure(figsize=(6.7, 5), dpi=100)
a = f.add_subplot(111)
a.grid(True)
#moment = time.strftime("%Y-%b-%d__%H_%M_%S",time.localtime())
#logging.basicConfig(filename='statistics_' + moment + '.log', level=logging.INFO)

def animate():
    global xListBest, xListAverage, yListBestFitness, yListAverageFitness, a
    a.clear()
    a.plot(xListBest, yListBestFitness, 'b-', label="Best")
    a.plot(xListAverage, yListAverageFitness, 'r-', label="Average")
    a.grid(True)
    a.get_xaxis().tick_bottom()
    a.get_yaxis().tick_left()
    a.set_xlabel('Number of Generations')
    a.set_ylabel('Cost')
    a.set_title('Population Evolution')
    f.canvas.draw()


class Gui(tk.Tk):
    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        label = tk.Label(self, text="Population Size", font=LARGE_FONT)
        label.grid(sticky=tk.W)
        label = tk.Label(self, text="Default: " + str(POPULATION_SIZE), font=SMALL_FONT)
        label.grid(sticky=tk.W)
        self.entry0 = tk.Entry(self, bd=5)
        self.entry0.grid(row=0, column=0)

        label = tk.Label(self, text="Tree Max Size", font=LARGE_FONT)
        label.grid(sticky=tk.W, row=0, column=1)
        label = tk.Label(self, text="Default: " + str(TREE_MAX_SIZE), font=SMALL_FONT)
        label.grid(sticky=tk.W, row=1, column=1)
        self.entry1 = tk.Entry(self, bd=5)
        self.entry1.grid(row=0, column=1)

        label = tk.Label(self, text="Number of Inputs", font=LARGE_FONT)
        label.grid(sticky=tk.W)
        label = tk.Label(self, text="Default: " + str(N_INPUTS), font=SMALL_FONT)
        label.grid(sticky=tk.W)
        self.entry2 = tk.Entry(self, bd=5)
        self.entry2.grid(row=2, column=0)

        label = tk.Label(self, text="Tournament Size", font=LARGE_FONT)
        label.grid(sticky=tk.W, row=2, column=1)
        label = tk.Label(self, text="Default:" + str(TOURNAMENT_SIZE), font=SMALL_FONT)
        label.grid(sticky=tk.W, row=3, column=1)
        self.entry3 = tk.Entry(self, bd=5)
        self.entry3.grid(row=2, column=1)

        label = tk.Label(self, text="Number of Generations", font=LARGE_FONT)
        label.grid(sticky=tk.W)
        label = tk.Label(self, text="Default: " + str(NUMBER_OF_GENERATIONS), font=SMALL_FONT)
        label.grid(sticky=tk.W)
        self.entry4 = tk.Entry(self, bd=5)
        self.entry4.grid(row=4, column=0)

        label = tk.Label(self, text="Mutation Probability", font=LARGE_FONT)
        label.grid(sticky=tk.W, row=4, column=1)
        label = tk.Label(self, text="Default: " + str(MUTATION_PROBABILITY), font=SMALL_FONT)
        label.grid(sticky=tk.W, row=5, column=1)
        self.entry5 = tk.Entry(self, bd=5)
        self.entry5.grid(row=4, column=1)

        button = tk.Button(self, text="Start", command=self.startGeneticProgramming, font=LARGE_FONT)
        button.grid(row=12)

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().grid(row=13, column=0)

        self.text = tk.Text(self, wrap="word")
        self.text.grid(row=13, column=1)

    def write(self, txt):
        self.text.insert(tk.END, str(txt))

    def startGeneticProgramming(self):
        #for index in range(1, 100):
            print self.entry1.get(), self.entry2.get(), self.entry3.get()
            global TREE_MAX_SIZE, POPULATION_SIZE, N_INPUTS, TOURNAMENT_SIZE, NUMBER_OF_GENERATIONS, MUTATION_PROBABILITY, lO, lT

            if self.entry0.get() != "":
                POPULATION_SIZE = int(self.entry0.get())
            if self.entry1.get() != "":
                TREE_MAX_SIZE = int(self.entry1.get())
            if self.entry2.get() != "":
                N_INPUTS = int(self.entry2.get())
            if self.entry3.get() != "":
                TOURNAMENT_SIZE = int(self.entry3.get())
            if self.entry4.get() != "":
                NUMBER_OF_GENERATIONS = int(self.entry4.get())
            if self.entry5.get() != "":
                MUTATION_PROBABILITY = float(self.entry5.get())

            PARITY_SIZE_M = 2 ** N_INPUTS

            inputs = [None] * PARITY_SIZE_M
            outputs = [None] * PARITY_SIZE_M

            for i in range(PARITY_SIZE_M):
                inputs[i] = [None] * N_INPUTS
                value = i
                dividor = PARITY_SIZE_M
                parity = 1
                for j in range(N_INPUTS):
                    dividor /= 2
                    if value >= dividor:
                        inputs[i][j] = 1
                        parity = int(not parity)
                        value -= dividor
                    else:
                        inputs[i][j] = 0
                outputs[i] = parity

            print outputs

            lO = createOperationList(2)
            lT = createTerminalList(N_INPUTS)
            combList = createCombList(N_INPUTS)

            print lO, lT, combList
            #logging.info("\n\nTree Max Size: " + str(TREE_MAX_SIZE) + ", " +
            #             "Populations Size: " + str(POPULATION_SIZE) + ", " +
            #             "Number of Inputs: " + str(N_INPUTS) + ", " +
            #             "Tournament Size: " + str(TOURNAMENT_SIZE) + ", " +
            #             "Number of Generations: " + str(NUMBER_OF_GENERATIONS) + ", " +
            #             "Mutation Probability: " + str(MUTATION_PROBABILITY) + ", " +
            #             "Start Time: " + str(datetime.datetime.now()))
            population = Population()
            population.initPopulation(POPULATION_SIZE, lO, lT, combList, outputs)
            population.sortArray()
            population.reproduction(NUMBER_OF_GENERATIONS, lO, combList, outputs, TOURNAMENT_SIZE)
            self.write("\nBEST INDIVIDUAL: \n" + str(population.array[0]['tree']) + "\nSIZE: " +
                       str(population.array[0]['tree'].getSize()) + "\nFITNESS: " + str(population.array[0]['fitness']))


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

    def secondMutation(self, coin, listOperations, listTerminals):
        # The node "not" can't be mutated, since it's the only one with arity = 1.
        tmpListOperations = listOperations[:]
        tmpListOperations.remove(['not', 1])
        nodes = [self]
        while 1:
            newNodes = []
            if len(nodes) == 0:
                break
            for node in nodes:
                if len(node.children) > 0:
                    for child in node.children:
                        newNodes.append(child)
                if random.random() <= coin:
                    if [node.value, len(node.children)] in tmpListOperations:
                        random_Operation = random.randrange(0, len(tmpListOperations))
                        node.value = tmpListOperations[random_Operation][0]
                    elif node.value in listTerminals:
                        random_Terminal = random.randrange(0, len(listTerminals))
                        node.value = listTerminals[random_Terminal]
                        node.children = []
            nodes = newNodes
        return

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
        fitness = 0.0
        evaluation = self.evaluateCircuit(combList)
        # check if the tree surpasses the maximum size allowed.
        if self.getSize() > TREE_MAX_SIZE:
            fitness = 1000.0
            return fitness
        # If the evaluation is smaller than the combList, something went wrong.
        elif len(combList) != len(evaluation):
            fitness = 1000.0
            return fitness
        elif self.getSize() < 2 * N_INPUTS:
            fitness = 1000.0
            return fitness
        # Adds 100 cost to each wrong evaluation result
        else:
            for i in range(len(S)):
                if S[i] != evaluation[i]:
                    fitness += 1.0
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
        terminalIndex = 0
        while True:
            decisionMaking = random.randint(1, 100)
            # If emptyTerminals equals zero, there're no more empty spaces to add nodes in the tree.
            if emptyTerminals == 0:
                break
            # The probability of adding a terminal needs to be bigger than the one to add a function.
            elif decisionMaking < 50:
                random_Operation = random.randrange(0, len(listOperations))
                # Add a new node to the tree, selected randomically from the operation list.
                if newNode.addBinaryNode(
                        BinaryNode(listOperations[random_Operation][0], listOperations[random_Operation][1], [])):
                    emptyTerminals += listOperations[random_Operation][1] - 1
            else:

                # Add a new node to the tree, selected randomically from the terminal list.
                if newNode.addBinaryNode(BinaryNode(listTerminals[terminalIndex], 0, [])):
                    emptyTerminals -= 1
                    terminalIndex += 1
                    if (terminalIndex == len(listTerminals)):
                        terminalIndex = 0

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

    def reproduction(self, cycles, lO, combList, S, tournamentSize):
        global xListBest, xListAverage, yListBestFitness, yListAverageFitness, POPULATION_SIZE
        count = 0
        difference = 1000000000.0
        xListBest = []
        xListAverage = []
        yListBestFitness = []
        yListAverageFitness = []
        average = 0.0
        crossCandidates = []

        #        while(difference > 0):
        while (count < cycles):
            del crossCandidates[:]
            # Select randomically a number of different individuals, and sort them.
            # The best two are going to reproduction.
            while len(crossCandidates) < tournamentSize:
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
            crossCandidates[0]['tree'].secondMutation(MUTATION_PROBABILITY, lO, lT)
            # Calculate new individual's fitness
            crossCandidates[0]['fitness'] = crossCandidates[0]['tree'].calcFitness(combList, S)

            # Select a random index to be performed the crossOver on the individual
            crossIndex = random.randint(1, crossCandidates[1]['tree'].getSize())
            # Cross Over
            crossCandidates[1]['tree'].crossOver(crossIndex, newdict['tree'])
            # Mutation
            crossCandidates[1]['tree'].secondMutation(MUTATION_PROBABILITY, lO, lT)
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

            # Animation control
            if count % 1000 == 0:
                average = 0.0
                for index in range(0, len(self.array)):
                    average += self.array[index]['fitness']
                average = average / POPULATION_SIZE
                xListBest.append(count)
                xListAverage.append(count)
                yListBestFitness.append(difference)
                yListAverageFitness.append(average)
                animate()

            elif count % 100 == 0:
                xListBest.append(count)
                yListBestFitness.append(difference)
                animate()

            # Did we find the desired result?
            if difference == 0 or count >= cycles:
                average = 0.0
                for index in range(0, len(self.array)):
                    average += self.array[index]['fitness']
                average = average / POPULATION_SIZE

                #logging.info("Best Fitness: " + str(difference) + ", " +
                #             "Best Size: " + str(bestSize) + ", " +
                #             "Average Fitness: " + str(average) + ", " +
                #             "Generations: " + str(count) + ", " +
                #             "Finish Time: " + str(datetime.datetime.now()) + "\n")

            print difference, bestSize, worst, worstSize, count

            if difference == 0:
                xListBest.append(count)
                yListBestFitness.append(difference)
                average = 0.0
                for index in range(0, len(self.array)):
                    average += self.array[index]['fitness']
                average = average / POPULATION_SIZE
                xListAverage.append(count)
                yListAverageFitness.append(average)
                animate()
                break

        return

    def reproductionWithSelection(self, cycles, lO, combList, S, tournamentSize):
        global xListBest, xListAverage, yListBestFitness, yListAverageFitness, POPULATION_SIZE, CROSS_PROB
        count = 0
        difference = 1000000000.0
        xListBest = []
        xListAverage = []
        yListBestFitness = []
        yListAverageFitness = []
        average = 0.0
        operations = 0
        offspringCandidates = []
        offspring = []
        #        while(difference > 0):
        while (count < cycles):
            del offspringCandidates[:]
            del offspring[:]
            # Select randomically a number of different individuals, and sort them.
            # The best two are going to reproduction.

            # SELECTION
            while len(offspring) < POPULATION_SIZE:
                del offspringCandidates[:]
                while len(offspringCandidates) < tournamentSize:
                    randomIndex = random.randrange(len(self.array))
                    # if self.array[randomIndex] not in crossCandidates:
                    offspringCandidates.append(self.array[randomIndex])
                offspringCandidates = self.sortSingleArray(offspringCandidates)[:]
                offspring.append(copy.deepcopy(offspringCandidates[0]))

            offspring = self.sortSingleArray(offspring)[:]

            for index in range(1, len(offspring)):
                if random.random() <= CROSS_PROB:
                    # Select a random index to be performed the crossOver on the individual
                    crossIndex = random.randint(1, offspring[index]['tree'].getSize())
                    # Cross Over
                    offspring[index]['tree'].crossOver(crossIndex, offspring[index-1]['tree'])
                    operations += 1
                # Mutation
                offspring[index]['tree'].secondMutation(MUTATION_PROBABILITY, lO, lT)
                # Calculate new individual's fitness
                offspring[index]['fitness'] = offspring[index]['tree'].calcFitness(combList, S)
            self.array = copy.deepcopy(offspring)
            self.sortArray()

            # Check the best individual of the population
            difference = self.array[0]['fitness']
            bestSize = self.array[0]['tree'].getSize()
            worst = self.array[len(self.array) - 1]['fitness']
            worstSize = self.array[len(self.array) - 1]['tree'].getSize()

            # Start a new generation
            count += 1

            # if count % 1000 == 0:
            #     average = 0.0
            #     for index in range(0, len(self.array)):
            #         average += self.array[index]['fitness']
            #     average = average / POPULATION_SIZE
            #     xListBest.append(count)
            #     xListAverage.append(count)
            #     yListBestFitness.append(difference)
            #     yListAverageFitness.append(average)
            #     animate()
            #
            # elif count % 100 == 0:
            #     xListBest.append(count)
            #     yListBestFitness.append(difference)
            #     animate()

            # Did we find the desired result?
            if difference == 0 or count >= cycles:
                average = 0.0
                for index in range(0, len(self.array)):
                    average += self.array[index]['fitness']
                average = average / POPULATION_SIZE

                logging.info("Best Fitness: " + str(difference) + ", " +
                             "Best Size: " + str(bestSize) + ", " +
                             "Average Fitness: " + str(average) + ", " +
                             "Generations: " + str(count) + ", " +
                             "Finish Time: " + str(datetime.datetime.now()) + "\n")

            print difference, bestSize, worst, worstSize, count, operations

            if difference == 0:
                # xListBest.append(count)
                # yListBestFitness.append(difference)
                # average = 0.0
                # for index in range(0, len(self.array)):
                #     average += self.array[index]['fitness']
                # average = average / POPULATION_SIZE
                # xListAverage.append(count)
                # yListAverageFitness.append(average)
                # animate()
                break

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


app = Gui()
app.mainloop()
