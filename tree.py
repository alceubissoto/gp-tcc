import sys, random, math, copy
import numpy as np
sys.setrecursionlimit(20000)
class BinaryNode(object):
    def __init__(self, value = None, arity = 2, children = [], size=0):
        self.value = value
        self.arity = arity
        self.children = children
        self.size = size

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
            return self.children[0].evaluate(x) / self.children[1].evaluate(x)
        else:
            return self.value

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
            self.size += 1
            return True
        elif self.arity > 0:
            for child in self.children:
                if child.addBinaryNode(binaryNode):
                    return True
        else:
            #print 'FALHOU: ', binaryNode.value
            return False

    def getSizeiii(self):
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


    def crossOver(self, counter, t2, counter2):
        if len(self.children) >= counter:
            if t2.selectNode(counter) is not None:
                print "INSIDE CROSS 1:", t2.selectNode(counter)
                self.children[counter-1] = t2.selectNode(counter2)
                return
            else:
                print "INSIDE CROSS 1/2: "
                self.children[counter-1] = t2
                return
        else:
            counter -= len(self.children)
            for child in self.children:
                print "INSIDE CROSS 2"
                if child.getSize() >= counter:
                    print "INSIDE CROSS 3"
                    return child.crossOver(counter, t2, counter2)
                else:
                    print "INSIDE CROSS 4"
                    counter -= child.getSize()
    

    def calcFitness(self, x, func):
        return np.sum(np.power(self.evaluate(x)-func,2))

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
        
    def appendRandomTree(self, listOperations, listTerminals):
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
        self.array.append({'fitness':newNode.calcFitness(x, func),'tree':newNode})

    def initPopulation(self, size, listOperations, listTerminals):
        for i in range(0, size):
            self.appendRandomTree(listOperations, listTerminals)
 
    def sortArray(self):
        tmp = sorted(self.array, key= lambda ind: ind['fitness'])
        self.array = tmp[:]

    def reproduction(self, kept):
        count = 0
        difference = 1000000000.0
        while(difference > 1.0):
            for i in range(1, len(self.array)):
                self.array.pop(i)
                self.appendRandomTree(lO, lT)

            self.sortArray()
            for i in range(0, kept):
                self.array.pop()
                newdict = copy.deepcopy(self.array[i])
                self.array.append(newdict)
               # print "TEMP DICT: ", newdict
                crossPartner = random.randint(i+1,len(self.array)-1)
               # print "CROSS PARTNER: ", crossPartner
                crossPartnerIndex = random.randint(2, self.array[crossPartner]['tree'].getSize())
                crossIndex = random.randint(2, self.array[i]['tree'].getSize())
               # print "CROSS PARTNER: ", crossPartner, "CROSSINDEX: ", crossPartnerIndex
               # print "BEFORE[",i,"] :",  self.array[i], "\nsize: ", self.array[i]['tree'].getSize()
               # print "PARTNER[",i,"] :", self.array[crossPartner], "\nsize: ", self.array[crossPartner]['tree'].getSize()
               # self.array[i]['tree'].crossOver(crossIndex, self.array[crossPartner]['tree'], crossPartnerIndex)
               # print "AFTER[",i,"] :", self.array[i], "\nsize: ", self.array[i]['tree'].getSize()
               # print "LASTPOSITION[",i,"] :", self.array[len(self.array)-1], "\nsize: ", self.array[len(self.array)-1]['tree'].getSize()
                self.sortArray()
            difference = self.array[0]['fitness']
            count += 1
#            for i in range(0, len(self.array)):
#                size += float(self.array[i]['tree'].getSize()+1)
#            size = size/float(len(self.array))
            print difference, count, len(self.array)
#            difference = 0            
        return count
                   
            
x = np.array([-1000.0,-500.0,-100.0,0.0,100.0,500.0,1000.0])
func = eval('x**2+x')
lO = [['+',2],['*',2],['-',2]]
lT = ['x', 1.0, 2.0]
lT2 = ['x', 3.0, 5.0]
#for i in range(1, 10):
#    newNode = generateRandomTree(lO, lT)
#    geno.append(newNode)

#for i in range(1, 10):
#    print geno[i-1], "with size: ", geno[i-1].size
population = Population()
population.initPopulation(5, lO, lT)
population.sortArray()
print "SORTED: ", population.array[0]['tree'], "SIZE: ", population.array[0]['size']
#print "SELECTED: ", population.array[0]['tree'].selectNode(3)
#print "SIZE: ", population.array[0]['tree'].iterativeChildren()
#print "GENERATIONS: ", population.reproduction(2)
#print "BEST INDIVIDUAL: ", population.array[0]
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
