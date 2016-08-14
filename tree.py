import sys, random, math, copy
import numpy as np
from itertools import chain, imap

sys.settrace
class BinaryNode(object):
    def __init__(self, value = None, arity = 2, children = []):
        self.value = value
        self.children = children
        self.arity = arity

    def __repr__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\n"
        for child in self.children:
            ret += child.__repr__(level+1)
        return ret

    def __iter__(self):
        for v in chain(*imap(iter, self.children)):
            yield v
        yield self.value

    def evaluate(self, x, listOperations):
        to_eval = list(iter(self))
        newlist = []
        while to_eval:

            tmp1 = to_eval.pop(0)
            tmp2 = to_eval.pop(0)
            tmp3 = to_eval.pop(0)

            if [tmp3, 2] in listOperations and [tmp1, 2] not in listOperations and [tmp2, 2] not in listOperations:
                newlist.append('(' + str(tmp1) + tmp3 + str(tmp2) + ')')
            else:
                newlist.append(tmp1)
                to_eval.insert(0,tmp3)
                to_eval.insert(0,tmp2)

            if len(to_eval) < 3 and len(to_eval) > 0:
                while to_eval:
                    newlist.append(to_eval.pop(0))
                to_eval = list(newlist)
                newlist = []

        return eval(newlist[0])


    def evaluateRec(self, x):
        if self.value == 'x':
            return x
        elif self.value == '+':
            return self.children[0].evaluate(x) + self.children[1].evaluate(x)
        elif self.value == '-':
            return self.children[0].evaluate(x) - self.children[1].evaluate(x)
        elif self.value == '*':
            return self.children[0].evaluate(x) * self.children[1].evaluate(x)
        elif self.value == '/':
            try: 
                return self.children[0].evaluate(x) / self.children[1].evaluate(x)
            except ZeroDivisionError: 
                return 1.0
        else:
            return float(self.value)

    def writeFunc(self, array):
        if len(self.children) == 2:
            array.append('(')
            self.children[0].writeFunc(array)
            array.append(self.value)
            self.children[1].writeFunc(array)
            array.append(')')
        else:
            array.append(self.value)
        return array
        
    def addBinaryNode(self, binaryNode):
        if len(self.children) < self.arity:
            self.children.append(binaryNode)
            return True
        elif self.arity > 0:
            for child in self.children:
                if child.addBinaryNode(binaryNode):
                    return True
        else:
            return False

    def getSizeI(self):
        counter = len(self.children)
        for child in self.children:
            counter += child.getSize()
        return counter

    def getSize(self):
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


    def selectNode(self, counter):
        if counter == 0:
            return self
        if len(self.children) >= counter:
            return self.children[counter-1]
        else:
            counter -= len(self.children)
            for child in self.children:
                if child.getSize() >= counter:
                    return child.selectNode(counter)
                else:
                    counter -= child.getSize()

    def selectRandomNode(self):
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
        toBeMutated = self.selectRandomNode()
        if [toBeMutated.value, len(toBeMutated.children)] in listOperations:
            random_Operation = random.randrange(0,len(listOperations))
            toBeMutated.value = listOperations[random_Operation][0]
        else:
            random_Terminal = random.randrange(0,len(listTerminals))
            toBeMutated.value = listTerminals[random_Terminal]
            toBeMutated.children = []


    def crossOver(self, counter, t2):
        if counter == 0:
            return
        elif len(self.children) >= counter:
            tmp = t2.selectRandomNode()
            self.children[counter-1] = newdict = copy.deepcopy(tmp)
            counter = 0
        else:
            counter -= len(self.children)
            for child in self.children:
                if child.getSize() >= counter:
                    return child.crossOver(counter, t2)
                else:
                    counter -= child.getSize()
        return
    

    def calcFitness(self, x, lO, func):
        return np.sum(np.power(self.evaluate(x, lO)-func,2))+self.getSize()
#        return np.sum(np.power(eval(''.join([str(y) for y in self.writeFunc([])]))-func, 2))

 
class Population(object):
    def __init__(self, array = []):
        self.array = array
        
    def appendRandomTree(self, listOperations, listTerminals, x, func):
        random_Operation = random.randrange(0,len(listOperations))
        newNode = BinaryNode(listOperations[random_Operation][0],listOperations[random_Operation][1], [])
        emptyTerminals = listOperations[random_Operation][1]
        while True:
            decisionMaking = random.randint(1, 100)
            if emptyTerminals == 0:
                break
            #The probability of adding a terminal needs to be bigger than the one to add a function.
            elif decisionMaking < 40:
               random_Operation = random.randrange(0,len(listOperations))
               if newNode.addBinaryNode(BinaryNode(listOperations[random_Operation][0], listOperations[random_Operation][1], [])):
                   emptyTerminals += listOperations[random_Operation][1] - 1
            else:
               random_Terminal = random.randrange(0,len(listTerminals))
               if newNode.addBinaryNode(BinaryNode(listTerminals[random_Terminal], 0, [])):
                   emptyTerminals -= 1
        if (newNode.getSize() > 100):
            return self.appendRandomTree(listOperations, listTerminals, x, func)
        self.array.append({'fitness':newNode.calcFitness(x, listOperations, func),'tree':newNode})

    def initPopulation(self, size, listOperations, listTerminals, x, func):
        for i in range(0, size):
            self.appendRandomTree(listOperations, listTerminals, x, func)
 
    def sortArray(self):
        tmp = sorted(self.array, key= lambda ind: ind['fitness'])
        self.array = tmp[:]

    def reproduction(self, kept, cycles, lO, x, func):
        count = 0
        difference = 1000000000.0
        while(count < cycles):
            for i in range(kept, len(self.array)):
                self.array.pop(i)
                self.appendRandomTree(lO, lT, x, func)
            self.sortArray()

            for i in range(0, kept):
                self.array.pop()

            for i in range(0, kept):
                newdict = copy.deepcopy(self.array[i])
                self.array.append(newdict)
                # CROSSOVER
                crossPartner = random.randint(i+1,len(self.array)-1)
                crossIndex = random.randint(2, self.array[i]['tree'].getSize())
                self.array[i]['tree'].crossOver(crossIndex, self.array[crossPartner]['tree'])
                self.array[i]['fitness'] = self.array[i]['tree'].calcFitness(x, lO, func)
            self.sortArray()     
            # MUTATION
            mutationSelection = random.randint(kept, len(self.array)-1)
            self.array[mutationSelection]['tree'].mutation(lO, lT)
            self.array[mutationSelection]['fitness'] = self.array[mutationSelection]['tree'].calcFitness(x, lO, func)
            self.sortArray()
            difference = self.array[0]['fitness']
            count += 1
            print difference, count
#            size = 0
#            difference = 0            
        return count
                   
x_array = np.array([-1000.0,-500.0,-100.0,0.0,100.0,500.0,1000.0])
func = eval('x_array**3+x_array**2+x_array+1')
lO = [['AND',2],['OR',2],['NOT',2]]
lT = ['x', 1.0, 2.0]
lT2 = ['x', 3.0, 5.0]

population = Population()
population.initPopulation(50, lO, lT, x_array, func)
#population.sortArray()
#size = 0
#print "SELECTED: ", population.array[0]['tree'].selectNode(3)
print "GENERATIONS: ", population.reproduction(10, 5000, lO, x_array, func)
print "BEST INDIVIDUAL: ", population.array[0], "SIZE: ", population.array[0]['tree'].getSize(), "FITNESS: ", population.array[0]['tree'].calcFitness(x_array, lO, func)
#print list(iter(population.array[0]['tree']))
#print population.array[0]['tree'].evaluateList(x_array, lO)
#print "writeFuncIt: ", population.array[0]['tree'].writeFuncIt()
#print "writeFunc: ", ''.join([str(x) for x in population.array[0]['tree'].writeFunc(arraywf)])
#print "eval: ", eval(''.join([str(x) for x in population.array[0]['tree'].writeFunc(arraywf)]))
#print "eval original: ", np.sum(np.power(population.array[0]['tree'].evaluate(x)-func, 2)) 
#population.array[0]['tree'].mutation(lO, lT)
#print "SIZE: ", population.array[0]['tree'].getSize()
#print "MUTATED: ", population.array[0]
