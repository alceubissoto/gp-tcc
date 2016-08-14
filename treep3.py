import random, math, copy
import numpy as np

class BinaryNode(object):
    def __init__(self, value = None, arity = 2, children = []):
        self.value = value
        self.arity = arity
        self.children = children

    def __repr__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\n"
        for child in self.children:
            ret += child.__repr__(level+1)
        return ret

    def evaluate(self, x):
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

    def flatten(self):
        left = []
        parents = []
        bst = self
        parents.append(bst)
        def descend_left(bst):
            while bst.children:
                parents.append(bst.children[0])
                parents.append(bst.children[1])
                bst = bst.children[0]
        descend_left(bst)
        while parents:
            bst = parents.pop()
            left.append(bst.value)
            if bst.children:
                descend_left(bst.children[1])
        return left

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
            #print 'ADICIONOU: ', binaryNode.value
            self.children.append(binaryNode)
            return True
        elif self.arity > 0:
            for child in self.children:
                if child.addBinaryNode(binaryNode):
                    return True
        else:
            #print 'FALHOU: ', binaryNode.value
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

    def evalWriteFunc(self, array, listOperations):
        indexOp = 0
        indexCh = 0
        sortedArray = []
        sortedArray.append(array.pop(indexOp))
        while indexCh < len(array):
            if [(array[indexOp]), 2] in listOperations:
                indexCh += 1
                sortedArray.insert(indexOp, array.pop(indexCh-1))
                sortedArray.insert(indexOp + 2, array.pop(indexCh))
 
            
            
        
        

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
        results = []
        nodes = self.children
        while 1:
            newNodes = []
            for node in nodes:
                counter -= 1
                if counter == 0:
                    return node
                #results.append(node.value)
                if len(node.children) > 0:
                    for child in node.children:
                        newNodes.append(child)
            nodes = newNodes


    def mutation(self, counter, listOperations, listTerminals):
        toBeMutated = self.selectNode(counter)
        if [toBeMutated.value, len(toBeMutated.children)] in listOperations:
            random_Operation = random.randrange(0,len(listOperations))
            print (listOperations[random_Operation][0])
            toBeMutated.value = listOperations[random_Operation][0]
        else:
            random_Terminal = random.randrange(0,len(listTerminals))
            toBeMutated.value = listTerminals[random_Terminal]
            toBeMutated.children = []


    def crossOver(self, counter, t2):
        if counter == 0:
            return
        elif len(self.children) >= counter:
            counter = 0
            self.children[counter-1] = t2.selectRandomNode()
        else:
            counter -= len(self.children)
            for child in self.children:
                if child.getSize() >= counter:
                    return child.crossOver(counter, t2)
                else:
                    counter -= child.getSize()
        return
    

    def calcFitness(self, x, func):
        return np.sum(np.power(self.evaluate(x)-func,2))
#        return np.sum(np.power(eval(''.join([str(y) for y in self.writeFunc([])]))-func, 2))

#def generateRandomTree(listOperations, listTerminals):
#    random_Operation = random.randrange(0,len(listOperations))
#    newNode = BinaryNode(listOperations[random_Operation][0],listOperations[random_Operation][1], [])
#    emptyTerminals = listOperations[random_Operation][1]
#    while True:
#        decisionMaking = random.randint(1, 100)
#        if emptyTerminals == 0:
#            break
#        #The probability of adding a terminal needs to be bigger than the one to add a function.
#        elif decisionMaking < 40:
#           random_Operation = random.randrange(0,len(listOperations))
#           if newNode.addBinaryNode(BinaryNode(listOperations[random_Operation][0], listOperations[random_Operation][1], [])):
#               emptyTerminals += listOperations[random_Operation][1] - 1
#        else:
#           random_Terminal = random.randrange(0,len(listTerminals))
#           if newNode.addBinaryNode(BinaryNode(listTerminals[random_Terminal], 0, [])):
#               emptyTerminals -= 1
##        print emptyTerminals
#    return newNode

    
class Population(object):
    def __init__(self, array = []):
        self.array = array
        
    def appendRandomTree(self, listOperations, listTerminals, x):
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
#            print emptyTerminals
#        if (newNode.getSize() > 100):
#            return self.appendRandomTree(listOperations, listTerminals, x)
        self.array.append({'fitness':newNode.calcFitness(x, func),'tree':newNode})

    def initPopulation(self, size, listOperations, listTerminals, x):
        for i in range(0, size):
            self.appendRandomTree(listOperations, listTerminals, x)
 
    def sortArray(self):
        tmp = sorted(self.array, key= lambda ind: ind['fitness'])
        self.array = tmp[:]

    def reproduction(self, kept, cycles, x):
        count = 0
        difference = 1000000000.0
        while(count < cycles):
            for i in range(1, len(self.array)):
                self.array.pop(i)
                self.appendRandomTree(lO, lT, x)

            self.sortArray()
            for i in range(0, kept):
                self.array.pop()
                newdict = copy.deepcopy(self.array[i])
                self.array.append(newdict)
                # CROSSOVER
                crossPartner = random.randint(i+1,len(self.array)-1)
               # print "CROSS PARTNER: ", crossPartner
                #crossPartnerIndex = random.randint(2, self.array[crossPartner]['tree'].getSize())
                crossIndex = random.randint(2, self.array[i]['tree'].getSize())
                #print "CROSS PARTNER: ", crossPartner, "CROSSINDEX: ", crossPartnerIndex
                #print "BEFORE[",i,"] :",  self.array[i], "\nsize: ", self.array[i]['tree'].getSize()
                #print "PARTNER[",i,"] :", self.array[crossPartner], "\nsize: ", self.array[crossPartner]['tree'].getSize()
                self.array[i]['tree'].crossOver(crossIndex, self.array[crossPartner]['tree'])
                self.array[i]['fitness'] = self.array[i]['tree'].calcFitness(x, func)
                #print "AFTER[",i,"] :", self.array[i], "\nsize: ", self.array[i]['tree'].getSize()
                #print "LASTPOSITION[",i,"] :", self.array[len(self.array)-1], "\nsize: ", self.array[len(self.array)-1]['tree'].getSize()
                
                # MUTATION
                # mutationSelection = random.randint(kept, len(self.array)-1)
                 
                self.sortArray()
            difference = self.array[0]['fitness']
            count += 1
            print (difference, count)
#            size = 0
#            difference = 0            
        return count
                   
x_array = np.array([-1000.0, -500.0, -100.0, 0.0, 100.0, 500.0, 1000.0])
func = eval('x_array**3+x_array**2+x_array+1')
lO = [['+', 2], ['*', 2], ['-', 2]]
lT = ['x', 1.0, 2.0]
lT2 = ['x', 3.0, 5.0]
#for i in range(1, 10):
#    newNode = generateRandomTree(lO, lT)
#    geno.append(newNode)

#for i in range(1, 10):
#    print geno[i-1], "with size: ", geno[i-1].size
population = Population()
population.initPopulation(50, lO, lT, x_array)
#population.sortArray()
size = 0
#print "SELECTED: ", population.array[0]['tree'].selectNode(3)
#print ("GENERATIONS: ", population.reproduction(10, 500, x_array))
print ("BEST INDIVIDUAL: ", population.array[0], "SIZE: ", population.array[0]['tree'].getSize(), "FITNESS: ", population.array[0]['tree'].calcFitness(x_array, func))
print ("FLATTEN: ", population.array[0]['tree'].flatten())
#print "writeFuncIt: ", population.array[0]['tree'].writeFuncIt()
#b = population.array[0]['tree'].writeFuncIt()
#b.reverse()
#print b
#print "writeFunc: ", ''.join([str(x) for x in population.array[0]['tree'].writeFunc(arraywf)])
#print "eval: ", eval(''.join([str(x) for x in population.array[0]['tree'].writeFunc(arraywf)]))
#print "eval original: ", np.sum(np.power(population.array[0]['tree'].evaluate(x)-func, 2)) 
#population.array[0]['tree'].mutation(2, lO, lT)
#print "SIZE: ", population.array[0]['tree'].getSize()
#print "MUTATED: ", population.array[0]
#population = []
#for i in range(0, 10):
#    new = generateRandomTree(lO, lT)
#    population.append({'fitness':np.sum(np.power(new.evaluate(x)-func,2)),'tree':new})
#    newlist = list(sorted(population, key=lambda ind: ind['fitness']))
#    population = list(newlist)
#    difference = population[0]['fitness']

#
#def reproduction(population, difference):
#    temp = []
#    count = 0
#    while(difference > 1.0):
#        for i in range(2, 10):
#            new = generateRandomTree(lO, lT)
#            population[i] = ({'fitness':np.sum(np.power(new.evaluate(x)-func,2)),'tree':new})
#        newlist = sorted(population, key=lambda ind: ind['fitness'])
#        #for i in range(0, len(population)-2):
#        #    temp.append({'fitness':newlist[i].get('fitness'),'tree':newlist[i].get('tree')})
#        temp = newlist[:]
#        population = newlist[:]
#        tempdict = dict([('fitness',temp[0].get('fitness')), ('tree',temp[0].get('tree'))])
#        print "TEMP DICT: ", tempdict
#        #temp = list(newlist)
#        #temp.pop()
#        #temp.pop()
#        print "BEFORE :", population[0], "\nsize: ", population[0]['tree'].getSize()
#        print "POPULATION[1] :", population[1], "\nsize: ", population[1]['tree'].getSize()
#        cross1Index = random.randint(1, population[1]['tree'].getSize())
#       # cross2Index = random.randint(1, population[3]['tree'].getSize())
#        population[0]['tree'].crossOver(cross1Index, population[1]['tree'])
#       # population[1]['tree'].crossOver(cross2Index, temp[3]['tree'])
#        print "cross1Index :", cross1Index
#       # print "cross2Index :", cross2Index
#        print "AFTER:", population[0], "\nsize: ", population[0]['tree'].getSize()
#       # print "POPULATION[1] :", population[1], "\nsize: ", population[1]['tree'].getSize()
#        temp.append(tempdict)
#       # temp.append({'fitness':population[0].get('fitness'),'tree':population[0].get('tree')})
#       # temp.append({'fitness':population[1].get('fitness'),'tree':population[1].get('tree')})
#       # print "TEMP:\n"
#        print "TEMP[8] :", temp[8]['tree'], "\nsize: ", temp[8]['tree'].getSize()
#       # print "TEMP[9] :", temp[9], "\nsize: ", temp[9]['tree'].getSize()
#        print "TEMP[0] :", temp[0], "\nsize: ", temp[0]['tree'].getSize()
#       # print "TEMP[1] :", temp[1], "\nsize: ", temp[1]['tree'].getSize()
#
#    #    print "cross1Index :", cross1Index
#    #    print "cross2Index :", cross2Index
#    #    print "POPULATION[0] :", population[0]['tree'], "\nsize: ", population[0]['tree'].getSize()
#    #    print "POPULATION[1] :", population[1], "\nsize: ", population[1]['tree'].getSize()
#    #    print "POPULATION[8] :", population[8], "\nsize: ", population[8]['tree'].getSize()
#    #    print "POPULATION[9] :", population[9], "\nsize: ", population[9]['tree'].getSize()
#        #newlist = list(sorted(temp, key=lambda ind: ind['fitness']))
#        #population = list(newlist)
##       difference = population[0]['fitness']
#        difference = 0
#        count += 1
##    print population, "COUNT: ", count

#reproduction(population, difference)
#    new=generateRandomTree(lO, lT)
#    almostthere = np.power(new.evaluate(x)-func,2)
#    print almostthere
#    difference = np.sum(almostthere)
#    print difference
#    count += 1
#    if difference < 0:
#        difference = 10 
#print new, "".join([str(a) for a in new.writeFunc([])]), "\nCounter: ", count, "\nDifference:", difference

#new = generateRandomTree(lO, lT) 
#new2 = generateRandomTree(lO, lT2)

#print "ORIGINAL: \n", new, "size: ", new.getSize(), "NOTA: ", new.evaluate(x)
#print "NEW BRANCH: \n", new2, "size: ", new2.getSize(), "NOTA: ", new2.evaluate(x)
#new.crossOver(3, new2)
#print "THE NEW TREE: \n", new, "\nsize: ", new.getSize(), "\nNOTA: ", new.evaluate(x), "\nFUNC: ", "".join([str(a) for a in new.writeFunc([])])
#print "Difference: ", np.sum((new.evaluate(x)-func)**2)
#node = BinaryNode('*', 2)
#node.addBinaryNode(BinaryNode(10, 0, []))
#node.addBinaryNode(BinaryNode('+', 2, []))
#node.addBinaryNode(BinaryNode(3, 0, []))
#node.addBinaryNode(BinaryNode(2, 0, []))

#print node
#print node.evaluate()
