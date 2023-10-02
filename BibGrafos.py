import numpy as np
from collections import deque
from random import randint
class Grafo:
    def __init__(self, arquivo,represent="matriz", graph=None):# represent indica como a matriz vai ser representada e graph o grafo em si, que pode ser a lista ou a matriz
        self.arquivo = arquivo
        self.represent = represent
        self.graph = graph
        if self.graph==None:
                if self.represent=="matriz":
                  # Função para ler o arquivo de entrada e criar a matriz de adjacência
                  with open(self.arquivo, 'r') as arquivo:
                      num_vertices = int(arquivo.readline().strip())
                      matriz_adjacencia = np.zeros((num_vertices, num_vertices), dtype=np.int8)

                      for linha in arquivo:
                          v1, v2 = map(int, linha.strip().split())
                          v1 -= 1
                          v2 -= 1
                          if v1 != v2:
                            matriz_adjacencia[v1][v2] = 1
                            matriz_adjacencia[v2][v1] = 1

                      self.graph = matriz_adjacencia
                elif self.represent=="lista":
                  # Função para ler o arquivo de entrada e criar a lista de adjacência
                  with open(self.arquivo, 'r') as arquivo:
                    num_vertices = int(arquivo.readline().strip())
                    lista_adjacencia = [[] for _ in range(num_vertices)]

                    for linha in arquivo:
                        v1, v2 = map(int, linha.strip().split())
                        v1 -= 1
                        v2 -= 1
                        if v1 != v2:
                          lista_adjacencia[v1].append(v2)
                          lista_adjacencia[v2].append(v1)
                          lista_adjacencia[v1].sort()
                          lista_adjacencia[v2].sort()

                    self.graph = lista_adjacencia
                else:
                   raise Exception("Esse formato de grafo não existe ou não é suportado.")
                
    def num_vert(self):
      with open(self.arquivo, 'r') as arquivo:
        a = int(arquivo.readline().strip())
      return a


    def num_ares(self):
      with open(self.arquivo, 'r') as arquivo:
        a = len(arquivo.readlines())
      return a-1


    def gr_min(self):
      minimo = np.inf
      if self.represent=="matriz":
        minimo = min(np.sum(self.graph, axis=1))

      else:
        for vertice in self.graph:
          if len(vertice)<minimo:
            minimo = len(vertice)
      return minimo

    def gr_max(self):
      maximo = 0
      if self.represent=="matriz":
         maximo = max(np.sum(self.graph, axis=1))

      else:
        for vertice in self.graph:
            if len(vertice)>maximo:
              maximo = len(vertice)
      return maximo


    def gr_med(self):
      media = 0
      if self.represent=="matriz":
        media = np.mean(np.sum(self.graph, axis=1))

      else:
        graus = []
        for vertice in self.graph:
          grau = len(vertice)
          graus.append(grau)
        media = np.mean(graus)

      return media


    def mediana_gr(self):
      mediana = 0
      if self.represent=="matriz":
        mediana = np.median(np.sort(np.sum(self.graph, axis=1)))

      else:
        graus = np.empty(self.num_vert(), dtype = int)
        v = 0
        for vertice in self.graph:
          grau = len(vertice)
          graus[v] = grau
          v += 1
        mediana = np.median(np.sort(graus))
      return mediana

    def bfs(self,start): #Retorna a árvore geradora, uma lista com elementos no formato (pai, filho, nível)
      queue = deque()
      visitado = [False] * self.num_vert()
      caminho = []
      level = np.empty(self.num_vert(), dtype = int) #Vetor auxiliar que armazena os níveis de cada vértice na árvore geradora
      queue.append(start)
      caminho.append(("raiz", start , 0)) 
      level[start-1] = 0
      if self.represent == "matriz":
        while queue:
            v = queue.popleft()
            visitado[v-1]="Explorado"
            ones = np.where(self.graph[v-1] == 1)[0] #Retorna os índices das colunas da linha v com 1s
            for w in ones:
                if visitado[w]==False:
                    caminho.append((v, w+1, level[v-1] + 1))
                    visitado[w]="Descoberto"
                    level[w] = level[v-1] + 1
                    queue.append(w+1)
        file = open("bfs.txt",'w')
        file.write("A árvore geradora, no formato (pai,filho,level), é: ")
        file.write(str(caminho))
        file.close
        return caminho
      else:
        while queue:
          v = queue.popleft()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w] == False:
                caminho.append((v,w+1,level[v-1]+1))
                visitado[w] = "Descoberto"
                level[w] = level[v-1] +1
                queue.append(w+1)
        file = open("bfs.txt",'w')
        file.write("A árvore geradora, no formato (pai,filho,level), é: ")
        file.write(str(caminho))
        file.close
        return caminho
                  

    def dfs(self,start): #Retorna a árvore geradora, uma lista com elementos no formato (pai, filho, nível)
      stack = deque()
      visitado = [False] * self.num_vert()
      caminho = []
      level = np.empty(self.num_vert(), dtype = int)

      stack.append(start)
      caminho.append(("raiz", start , 0))
      level[start-1] = 0


      if self.represent == "matriz":
        while stack:
            v = stack.pop()
            visitado[v-1]="Explorado"
            ones = np.where(self.graph[v-1] == 1)[0] #Retorna os índices das colunas da linha v com 1s
            for w in ones:
                if visitado[w]==False:
                    caminho.append((v, w+1, level[v-1] + 1))
                    visitado[w]="Descoberto"
                    level[w] = level[v-1] + 1
                    stack.append(w+1)
        file = open("dfs.txt",'w')
        file.write("A árvore geradora, no formato (pai,filho,level), é: ")
        file.write(str(caminho))
        file.close
        return caminho
      else:
        while stack:
          v = stack.pop()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w] == False:
              caminho.append((v, w+1, level[v-1] + 1))
              visitado[w] = "Descoberto"
              stack.append(w+1)
              level[w] = level[v-1]+1
        file = open("dfs.txt",'w')
        file.write("A árvore geradora, no formato (pai,filho,level), é: ")
        file.write(str(caminho))
        file.close
        return caminho
    
    def distancia(self,v1,v2):
      queue = deque()
      visitado = [False] * self.num_vert()
      level = np.empty(self.num_vert(), dtype = int)
      queue.append(v1)
      level[v1-1] = 0
      if self.represent == "matriz":
        while queue:
            v = queue.popleft()
            visitado[v-1]="Explorado"
            ones = np.where(self.graph[v-1] == 1)[0]
            for w in ones:
                if visitado[w]==False:
                    visitado[w]="Descoberto"
                    level[w] = level[v-1] + 1
                    queue.append(w+1)
                    if w+1 == v2:
                       return level[w]
        return "Não há conexão entre os vértices dados"
      else:
        while queue:
          v = queue.popleft()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w] == False:
                visitado[w] = "Descoberto"
                level[w] = level[v-1] +1
                queue.append(w+1)
                if w+1 == v2:
                   return level[w]
        return "Não há conexão entre os vértices dados"

    def diametro(self): #Roda uma BFS em cada uma das vértices, retorna a maior distância possível
      diam = 0
      for start in range(1,self.num_vert()+1):
            queue = deque()
            visitado = [False] * self.num_vert()
            level = np.zeros(self.num_vert(), dtype = int)
            queue.append(start)
            level[start-1] = 0
            while queue:
                  v = queue.popleft()
                  visitado[v-1]="Explorado"
                  ones = np.where(self.graph[v-1] == 1)[0]
                  for w in ones:
                      if visitado[w]==False:
                          visitado[w]="Descoberto"
                          level[w] = level[v-1] + 1
                          queue.append(w+1)
            if np.max(level) > diam:
                diam = np.max(level)
      else:
          for start in range(1,self.num_vert()+1):
            queue = deque()
            visitado = [False] * self.num_vert()
            level = np.zeros(self.num_vert(), dtype = int)
            queue.append(start)
            level[start-1] = 0
            while queue:
              v = queue.popleft()
              visitado[v-1] = "Explorado"
              for w in self.graph[v-1]:
                if visitado[w] == False:
                    visitado[w] = "Descoberto"
                    level[w] = level[v-1] +1
                    queue.append(w+1)
            if np.max(level) > diam:
                diam = np.max(level)
      return diam
    
    def __bfs_v__(self,start,visitado): #BFS auxiliar que não armazena a árvore geradora
      queue = deque()
      componentes = [start]
      queue.append(start)
      if self.represent == "matriz":
        while queue:
            v = queue.popleft()
            visitado[v-1]="Explorado"
            ones = np.where(self.graph[v-1] == 1)[0] #Retorna os índices das colunas da linha v com 1s
            for w in ones:
                if visitado[w]==False:
                    componentes.append(w+1)
                    visitado[w]="Descoberto"
                    queue.append(w+1)
        return componentes
      else:
        while queue:
          v = queue.popleft()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w] == False:
                componentes.append(w+1)
                visitado[w] = "Descoberto"
                queue.append(w+1)
        return componentes
      
    def conexas(self): #Retorna as informações sobre as componentes conexas no formato (número de componentes, tamanhos das componentes, lista das componentes)
      visitado = [False] * self.num_vert()
      conexas = []
      comp_size = []
      z = 1 
      for x in visitado:
        if x == False:
          conexas.append(self.__bfs_v__(z,visitado))
        z += 1
      n_componentes = len(conexas)
      conexas.sort(key=len, reverse=True)
      for y in conexas:
         comp_size.append(len(y))
      return (n_componentes,comp_size,conexas)

    def info(self): #Gera um arquivo com várias informações
       c = self.conexas()
       file = open("grafo_info.txt",'w')
       file.write("O grafo contém ")
       file.write(str(self.num_vert()))
       file.write(" vértices e ")
       file.write(str(self.num_ares()))
       file.write(" arestas.\n")
       file.write("Seu grau mínimo é ")
       file.write(str(self.gr_min()))
       file.write(", seu grau máximo é ")
       file.write(str(self.gr_max()))
       file.write(", seu grau médio é ")
       file.write(str(self.gr_med()))
       file.write(" e sua mediana de grau é ")
       file.write(str(self.mediana_gr()))
       file.write(".\nHá ")
       file.write(str(c[0]))
       file.write(" componentes conexas, com tamanhos ")
       file.write(str(c[1]))
       file.write(".\nA lista de componentes conexas é ")
       file.write(str(c[2]))
       file.write(".")
       file.close()


    def diametro_a(self): #Diâmetro aproximado para grafos muito grandes
      start = randint(1,self.num_vert())
      queue = deque()
      visitado = [False] * self.num_vert()
      level = np.zeros(self.num_vert(), dtype = int)
      queue.append(start)
      level[start-1] = 0
      if self.represent == "matriz":
          while queue:
              v = queue.popleft()
              visitado[v-1]="Explorado"
              ones = np.where(self.graph[v-1] == 1)[0]
              for w in ones:
                  if visitado[w]==False:
                      visitado[w]="Descoberto"
                      level[w] = level[v-1] + 1
                      queue.append(w+1)
          diam = np.max(level)
          new_start = np.where(level == diam)[0][0]
      else:
          while queue:
            v = queue.popleft()
            visitado[v-1] = "Explorado"
            for w in self.graph[v-1]:
              if visitado[w] == False:
                  visitado[w] = "Descoberto"
                  level[w] = level[v-1] +1
                  queue.append(w+1)
          diam = np.max(level)
          new_start = np.where(level == diam)[0][0]
      queue = deque()
      visitado = [False] * self.num_vert()
      level = np.zeros(self.num_vert(), dtype = int)
      queue.append(new_start)
      level[new_start-1] = 0
      if self.represent == "matriz":
          while queue:
              v = queue.popleft()
              visitado[v-1]="Explorado"
              ones = np.where(self.graph[v-1] == 1)[0]
              for w in ones:
                  if visitado[w]==False:
                      visitado[w]="Descoberto"
                      level[w] = level[v-1] + 1
                      queue.append(w+1)
          if np.max(level) > diam:
             return np.max(level)
          else:
             return diam
      else:
          while queue:
            v = queue.popleft()
            visitado[v-1] = "Explorado"
            for w in self.graph[v-1]:
              if visitado[w] == False:
                  visitado[w] = "Descoberto"
                  level[w] = level[v-1] +1
                  queue.append(w+1)
          if np.max(level) > diam:
             return np.max(level)
          else:
             return diam
