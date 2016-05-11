import random, math
import numpy as np

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

    def getSize(self):
        counter = len(self.children)
        for child in self.children:
            counter += child.getSize()
        return counter

    def selectNode(self, counter, t2):
        if counter == 0:
            return self
        else:
            for child in self.children:
                if child.getSize()+1 >= counter:
                    return child.selectNode(counter-1, t2)
                else:
                    counter -= child.getSize()+1


    def crossOver(self, counter, t2):
        if counter == 0:
            return
        elif len(self.children) >= counter:
           self.children[counter-1] = t2
        else:
            counter -= len(self.children)
            for child in self.children:
                if child.getSize() >= counter:
                    return child.crossOver(counter, t2)
                else:
                    counter -= child.getSize()
        



def generateRandomTree(listOperations, listTerminals):
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
#        print emptyTerminals
    return newNode

    


x = np.array([-1000.0,-500.0,-100.0,0.0,100.0,500.0,1000.0])
func = eval('x**3+2*x**2+x+122')
lO = [['+',2],['*',2],['-',2]]
lT = ['x', 1.0, 2.0]
lT2 = ['x', 3.0, 5.0]
#for i in range(1, 10):
#    newNode = generateRandomTree(lO, lT)
#    geno.append(newNode)

#for i in range(1, 10):
#    print geno[i-1], "with size: ", geno[i-1].size
difference = 10.0
count =0
population = []
while(difference > 5.0):
    for i in range(1, 10):
        new = generateRandomTree(lO, lT)
        population.append({'fitness':new.evaluate(x),'tree':new})
    difference = 0

print population[0]
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
