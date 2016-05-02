class BinaryNode(object):
    def __init__(self, value = None, arity = 0, children = []):
        self.value = value
        self.arity = arity
        self.children = children

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
            print 'ADICIONOU'
            self.children.append(binaryNode)
            return 1
        elif self.arity > 0:
            counter = 0
            for child in self.children:
                counter = counter + 1
                if child.addBinaryNode(binaryNode):
                    print 'success, child added on counter: ', counter
                    break
        else:
            print 'FALHOU'
            return 0
       

  
node = BinaryNode('*', 2)
node.addBinaryNode(BinaryNode(10, 0, []))
node.addBinaryNode(BinaryNode('+', 2, []))
node.addBinaryNode(BinaryNode(3, 0, []))
node.addBinaryNode(BinaryNode(2, 0, []))

print node
print node.evaluate()
