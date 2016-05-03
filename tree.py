import random

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

    def evaluate(self):
        if self.value == '+':
            return self.children[0].evaluate() + self.children[1].evaluate()
        elif self.value == '-':
            return self.children[0].evaluate() - self.children[1].evaluate()
        elif self.value == '*':
            return self.children[0].evaluate() * self.children[1].evaluate()
        elif self.value == '/':
            return self.children[0].evaluate() / self.children[1].evaluate()
        else:
            return self.value

    def addBinaryNode(self, binaryNode):
        if len(self.children) < self.arity:
            print 'ADICIONOU: ', binaryNode.value
            self.children.append(binaryNode)
            self.size += 1
            return True
        elif self.arity > 0:
            for child in self.children:
                if child.addBinaryNode(binaryNode):
                    return True
        else:
            print 'FALHOU: ', binaryNode.value
            return False


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
        print emptyTerminals
    try:
        newNode.evaluate()
    except ZeroDivisionError:
        return generateRandomTree(listOperations, listTerminals)
    return newNode

def getSizeTree(t1):
    counter = len(t1.children)
    for child in t1.children:
        counter += getSizeTree(child)
    return counter

def selectNode(t1, counter):
    if counter == 0:
        return t1
    else:
       for child in t1.children:
           if getSizeTree(child)+1 >= counter:
               return selectNode(child, counter-1)
           else:
              counter -= getSizeTree(child)+1

lO = [['+',2],['*',2],['-',2],['/',2]]
lT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

geno = []
#for i in range(1, 10):
#    newNode = generateRandomTree(lO, lT)
#    geno.append(newNode)

#for i in range(1, 10):
#    print geno[i-1], "with size: ", geno[i-1].size



new = generateRandomTree(lO, lT) 
print new, "size: ", getSizeTree(new)
print selectNode(new, 3)
#print 'EVALUATION: ', new.evaluate()
#node = BinaryNode('*', 2)
#node.addBinaryNode(BinaryNode(10, 0, []))
#node.addBinaryNode(BinaryNode('+', 2, []))
#node.addBinaryNode(BinaryNode(3, 0, []))
#node.addBinaryNode(BinaryNode(2, 0, []))

#print node
#print node.evaluate()
