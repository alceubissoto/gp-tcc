import random
import numpy as np
from itertools import product
IND_MAX_SIZE = 10000
POP_SIZE = 5
MUTATION_CHANCE = 0.05
MAX_GENERATIONS = 100000
l_op = ['np.logical_and','np.logical_or','not np.logical_and','not np.logical_or','np.logical_xor', 'not np.logical_xor']
N_INPUTS = 8
PARITY_SIZE_M = 2 ** N_INPUTS

pop = [] #lista de genotipos

def createCombList(number_inputs):
    return ["".join(seq) for seq in product("01", repeat=number_inputs)]

def createTerminalList(number_inputs):
    lT = []
    for i in range(number_inputs):
        lT.append('in' + str(i))
    return lT

def createParityAnswer():
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
    return outputs

def evaluateIndividual(used_nodes, output):
        result = list()
        i = 0
        input_values = []
        # For each combination of 1s and 0s, evaluate:
        for combination in combList:
            for j in range(len(combination)):
                # Prepare the array A to pass the correct value of combination.
                # Example: (0, 0, 0)
                #         (0, 0, 1) ...
                input_values.append(int(combination[j]))
            # evaluateRec(A) is responsible for do the actual evaluation, with the A values passed that were crafted.
            result.append(decode(used_nodes, input_values, output))
            input_values[:] = []
            i += 1
        return result

def generateInd():
    new_ind = {}
    new_ind['genotype'] = []
    possible_values = list(range(0, len(l_in)))
    operators = list(range(0, len(l_op)))
    last = len(l_in)-1;
    for i in range(0, random.randrange(1,IND_MAX_SIZE)):
        new_ind['genotype'].append(random.choice(possible_values))
        new_ind['genotype'].append(random.choice(possible_values))
        new_ind['genotype'].append(random.choice(operators))
        last+=1
        possible_values.append(last)
    new_ind['output'] = random.choice(possible_values[N_INPUTS:])
    return new_ind

def selectUsedNodes(ind):
    out = ind['output']
    gen = ind['genotype']
    to_eval=[]
    to_eval.append(out)
    evaluated = list(range(0, len(l_in)))
    used_nodes = {}
    inicial_position = len(l_in)
    while(len(to_eval)>0):
        #print("to_eval", to_eval)
        if(to_eval[0] in evaluated):
            to_eval.pop(0)
        else:
            # node, (como obter seu valor (gen, gen, gen))
            #used_nodesput.append(to_eval[0])
            tmp = []
            for i in range(0,3):
                value = gen[(to_eval[0]-inicial_position)*3 + i]
                tmp.append(value)
                if((value not in evaluated) and (i!=2)):
                    to_eval.append(value)
            used_nodes[to_eval[0]] = tmp
            evaluated.append(to_eval[0])
    return used_nodes

def decode(used_nodes, input_values, output):
    tmp = []
    iterations = int(len(used_nodes))
    l_known = {}
    
    #inputs
    for i in range(0, len(input_values)):
        l_known[i]=bool(input_values[i])
        
    #evaluations
    for key in sorted(used_nodes.keys()):
        #l_known[key] = eval(str(l_known[used_nodes[key][0]])+' '+str(l_op[used_nodes[key][2]])\
        #                    +' '+str(l_known[used_nodes[key][1]]))
        l_known[key] = eval(str(l_op[used_nodes[key][2]])+'('+str(l_known[used_nodes[key][0]])\
                           +', '+str(l_known[used_nodes[key][1]])+')')
    #print(l_known)
    return l_known[output]

def calcFitness(used_nodes, output_node):
        fitness = 0.0
        evaluation = evaluateIndividual(used_nodes, output_node)
        #print(evaluation)
        for i in range(0, len(answer)):
            if answer[i] == evaluation[i]:
                fitness += 1.0
        return fitness

def createPopulation():
    population = []
    for i in range(0, POP_SIZE):
        temp_ind = generateInd()
        used_nodes = selectUsedNodes(temp_ind)
        temp_ind['fitness'] = calcFitness(used_nodes, temp_ind['output'])
        population.append(temp_ind)
    return population
        
def sortPopulation(population):
    newlist = sorted(population, key=lambda k: k['fitness'], reverse=True) 
    return newlist

def mutate(individual):
    possible_values = list(range(0, int(len(individual['genotype'])/3)))
    operators = list(range(0, len(l_op)))
    ind = individual
    active_nodes = selectUsedNodes(individual)
    new_ind={}
    mutated_genotype = []
    index = 0
    # Mutate Genotype
    #print('IND', ind)
    node_to_change = random.choice(list(active_nodes.keys()))
    gene_to_change = random.randint(0, 2)
    #print('ntc: ', node_to_change)
    #print('gtc: ', gene_to_change)
    which_gene = -1
    for i in range(0, len(ind['genotype'])):
        #print(ind['genotype'][i])
        #print('ifclause: ', int(i/3)+N_INPUTS)
        #print('which_gene', which_gene)
        if (int(i/3)+N_INPUTS == node_to_change):
            which_gene+=1
            if(which_gene == gene_to_change):
                if(gene_to_change==2):
                    #print('ntc: ', node_to_change)
                    #print('gtc: ', gene_to_change)
                    value_op = random.choice(operators)
                    #print('valueop', value_op)
                    mutated_genotype.append(value_op)
                else:
                    #print('ntc: ', node_to_change)
                    #print('gtc: ', gene_to_change)
                    value = random.choice(possible_values[0:int(i/3)+N_INPUTS])
                    #print('value', value)
                    mutated_genotype.append(value)
                which_gene=1000
            else:
                mutated_genotype.append(ind['genotype'][i])
        else:
            if(random.random() < MUTATION_CHANCE):
                if((i+1)%3 == 0):
                    mutated_genotype.append(random.choice(operators))
                else:
                    mutated_genotype.append(random.choice(possible_values[0:int((i+1)/3)+N_INPUTS]))
            else:
                mutated_genotype.append(ind['genotype'][i])
    
    new_ind['genotype'] = mutated_genotype
    # Mutate Output
    if(random.random() < MUTATION_CHANCE):
        new_ind['output'] = random.choice(possible_values)
    else:
        new_ind['output'] = individual['output']
    #print('NEW IND', new_ind)
    # Calculate new Fitness
    used_nodes = selectUsedNodes(new_ind)
    #print('output_m', individual['output'])
    #print('output_new', new_ind['output'])
    #print('un', used_nodes)
    #print('mutated', new_ind['genotype'])
    #print('before', individual['genotype'])
    new_ind['fitness'] = calcFitness(used_nodes, new_ind['output'])
    
    return new_ind

def reproduction():
    pop = createPopulation()
    sorted_pop = sortPopulation(pop)
    for i in range(0, MAX_GENERATIONS):
        new_pop=[]
        new_pop.append(sorted_pop[0])
        for j in range(1, POP_SIZE):
            new_pop.append(mutate(sorted_pop[0]))
        sorted_pop = sortPopulation(new_pop)
        print('gen: ', i, ', fit: ', sorted_pop[0]['fitness'])
        #print(sorted_pop)
        if (sorted_pop[0]['fitness']==PARITY_SIZE_M):
            print('generations to success: ', i)
            break
    return sorted_pop[0]
    
    
l_in = createTerminalList(N_INPUTS)    
combList = createCombList(N_INPUTS)
answer = createParityAnswer()
#print('answer', answer)

#new_ind = generateInd()
#used_nodes = selectUsedNodes(new_ind) #selectUsedNodes
#print(new_ind)
#node_to_change = random.choice(list(used_nodes.keys()))
#print('used_nodes', used_nodes)
#print('keys', used_nodes.keys())
#print(node_to_change)
#print('sorted', sorted(used_nodes.keys()))
#print('output', new_ind['output'])
#decode(used_nodes, new_ind['output'])
#print('evaluation', evaluateIndividual(combList, used_nodes, new_ind['output']))

#print('fitness', calcFitness(used_nodes, new_ind['output']))

#population = createPopulation()
#sortedpop = sortPopulation(population)
#print('new_ind1', new_ind['genotype'])
#mutated = mutate(new_ind)
#print('new_ind2', new_ind['genotype'])
#print('mutated1', mutated['genotype'])
#print('mutated', mutated)

best_individual = reproduction()
print(best_individual)
        
    

#l_op = ['and','or','xor']

