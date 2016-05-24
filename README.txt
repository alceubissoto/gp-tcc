Idéias gerais: 

O algoritmo usa-se de um array contendo os valores de x para quais serão calculados f(x);
uma lista de operações válidas (listOperations) contendo a operação e sua aridade: [ ['*',2], ['+',2] ] ;
uma lista de terminais válidos: [1,2,3,4,5,x] ;
e a função (func) que deseja se aproximar: (x³ + x² - x +1)

Binary node é uma classe que define um nó de uma árvore, responsável por representar uma função. Todo nó, portanto, tem um valor, sendo este uma função (operação matemática), ou um terminal (variável ou número real). Outro atributo é uma lista (do tamanho de sua aridade) de filhos. Sendo assim, é possível acessar uma árvore completa dado um binary node, já que há acesso aos seus filhos, que também são binary nodes.
Dentro desta classe, há alguns métodos que se aplicam a todos os nodes. Vamos a alguma delas:
evaluate(self, x): calcula, baseado em um array_x que diz em quais pontos quer se calcular x, o f(x). Foram desenvolvidas versões recursivas e iterativas de evaluate, assim como de algumas outras funções internas de modo a avaliar a performance dos métodos.

repr(self, level=0) mostra no console como se configura a árvore utilizando-se de tabs (\t). Ex:

getSize(self) calcula a quantidade de nós presentes na árvore. 

selectRandomNode(self) retorna uma subárvore aleatória válida de self.

crossOver(self, counter, t2): retorna modifica a árvore passada como parâmetro(self), de modo a fazer um crossOver com uma subárvore de t2. Figura:

mutation(self, listOperations, listTerminals): modifica o valor de um nó por um valor válido. Ou seja, substitui valores de operações (+, -, *) por outro do mesmo tipo. O mesmo para terminais.

calcFitness(self, func, x) calcula o fitness da árvore de acordo com a sua proximidade com a função "func" a qual se busca aproximar. O método utilizado é que o custo aumenta de acordo com a soma dos quadrados das diferenças entre os pontos da curva - Somatória( (xi1 - x01)² + (xi2 - x02)² + ... + (xin - x0n)² ). - mais o tamanho do indivíduo, de forma a incentivar que funções mais concisas sejam encontradas.


Uma população é composta por "size" indivíduos, e é inicializada por initPopulation(self, size, listOperations, listTerminals). Essa função é responsável por criar um array de BinaryNodes, ou seja, de árvores aleatórias - árvores essas criadas por appendRandomTree.
sortArray(self) organiza esse array de acordo com o fitness de cada árvore. O indivíduo mais apto ocupa a posição 0 (zero) do vetor.
reproduction faz manipulações nesse array, fazendo com que crossovers e mutações ocorram, por um determinado número de ciclos (cycles), mantendo os k (kept) individuos mais aptos da população todo ciclo.

