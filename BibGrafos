import numpy as np
from collections import deque
import sys
import random
import copy
import heapq
import time
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
                elif self.represent == "pesos":
                  with open(self.arquivo, 'r') as arquivo:
                    num_vertices = int(arquivo.readline().strip())
                    lista_adjacencia = [[] for _ in range(num_vertices)]

                    for linha in arquivo:
                        v1, v2, p = linha.strip().split()
                        v1,v2,p = int(v1),int(v2),float(p)
                        v1 -= 1
                        v2 -= 1
                        if v1 != v2:
                          lista_adjacencia[v1].append((v2,p))
                          lista_adjacencia[v2].append((v1,p))
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

    #A representação do grafo é importante para descobrir arestas entre vertices
    def bfs(self,start): #Retorna a árvore geradora, uma lista com elementos no formato (pai, filho, nível)
      queue = deque()
      visitado = [False] * self.num_vert()
      caminho = []
      level = np.empty(self.num_vert(), dtype = int) #Vetor auxiliar que armazena os níveis de cada vértice na árvore geradora
      queue.append(start)
      caminho.append(("raiz", start, 0))
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
      elif self.represent == "lista":
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
      else:
        while queue:
          v = queue.popleft()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w[0]] == False:
                caminho.append((v,w[0]+1,level[v-1]+1))
                visitado[w[0]] = "Descoberto"
                level[w[0]] = level[v-1] +1
                queue.append(w[0]+1)
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
      caminho.append(("raiz", start, 0))
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
      elif self.represent == "lista":
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
      else:
        while stack:
          v = stack.pop()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w[0]] == False:
              caminho.append((v, w[0]+1, level[v-1] + 1))
              visitado[w[0]] = "Descoberto"
              stack.append(w[0]+1)
              level[w[0]] = level[v-1]+1
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
      elif self.represent == "lista":
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
      else:
        while queue:
          v = queue.popleft()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w[0]] == False:
                visitado[w[0]] = "Descoberto"
                level[w[0]] = level[v-1] +1
                queue.append(w[0]+1)
                if w[0]+1 == v2:
                   return level[w[0]]
        return "Não há conexão entre os vértices dados"

    def diametro(self): #Roda uma BFS em cada uma das vértices, retorna a maior distância possível
      diam = 0
      if self.represent == "matriz":
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
          return diam
      elif self.represent == "lista":
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
                if visitado[w[0]] == False:
                    visitado[w[0]] = "Descoberto"
                    level[w[0]] = level[v-1] +1
                    queue.append(w[0]+1)
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
            # Ainda precisa verificar a aresta com v
            for w in ones:
                if visitado[w]==False:
                    componentes.append(w+1)
                    visitado[w]="Descoberto"
                    queue.append(w+1)
        return componentes
      elif self.represent == "lista":
        while queue:
          v = queue.popleft()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w] == False:
                componentes.append(w+1)
                visitado[w] = "Descoberto"
                queue.append(w+1)
        return componentes
      else:
        while queue:
          v = queue.popleft()
          visitado[v-1] = "Explorado"
          for w in self.graph[v-1]:
            if visitado[w[0]] == False:
                componentes.append(w[0]+1)
                visitado[w[0]] = "Descoberto"
                queue.append(w[0]+1)
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

    def diametro_a(self):
      c = self.conexas()
      diam = 0
      for x in c[2]:
        s = random.choice(x)
        b = self.bfs(s)
        if b[len(b)-1][2] > diam:
            diam = b[len(b)-1][2]
        b2 = self.bfs(b[len(b)-1][1])
        if b2[len(b2)-1][2] > diam:
            diam = b2[len(b2)-1][2]
      return diam

    def info(self): #Gera um arquivo com várias informações
       c = self.conexas()
       file = open("grafo_info.txt",'w')
       file.write("O grafo contém ")
       file.write(str(self.num_vert()))
       file.write(" vértices e ")
       file.write(str(self.num_ares()))
       file.write(" arestas.\n")
       file.write("Seu grau mímino é ")
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
    
    def vector_dijkstra_all(self,start):
      dist = [np.inf] * self.num_vert()
      descoberto = [False] * self.num_vert()
      fila = [start-1]
      dist[start-1] = 0
      caminho = [[start] for x in range(0,self.num_vert())]
      while fila != []:
        dist_u = np.inf
        for x in fila:
              if dist_u> dist[x]:
                u = x
                dist[u] = dist[x]
                dist_u = dist[x]
        for v,weight in self.graph[u]:
                if descoberto[v] == False:
                  fila.append(v)
                  descoberto[v] = True
                if weight < 0:
                  raise Exception("Esta biblioteca não implementa caminhos mínimos em grafos com pesos negativos")
                if dist[v]>dist[u]+ weight:
                  caminho[v] = copy.deepcopy(caminho[u])
                  dist[v]=dist[u]+ weight
                  caminho[v].append(v+1)
        fila.remove(u)
      return (dist,caminho)

    def vector_dijkstra(self,start,end):
      djk = self.vector_dijkstra_all(start)
      return (djk[0][end-1],djk[1][end-1])
    
    def heap_dijkstra_all(self,start):
        dist = [np.inf] * self.num_vert()
        # Inicialize a distância para o vértice de origem como 0
        dist[start-1] = 0
        # Fila de prioridade para manter os vértices não explorados com distâncias mínimas
        priority_queue = [(0, start-1)]
        caminho = [[start] for x in range(0,self.num_vert())]
        while priority_queue:
              # Escolhe o vértice com a menor distância
              dist_u, u = heapq.heappop(priority_queue)
              # Explore o vértice
              for (v, weight) in self.graph[u]:
                  if weight < 0:
                    raise Exception("Esta biblioteca não implementa caminhos mínimos em grafos com pesos negativos")
                  distance = dist_u + weight
                  if distance < dist[v]:
                      dist[v] = distance
                      heapq.heappush(priority_queue, (distance, v))
                      caminho[v] = copy.deepcopy(caminho[u])
                      caminho[v].append(v+1)
        return (dist,caminho)
    
    def heap_dijkstra(self,start,end):
      djk = self.heap_dijkstra_all(start)
      return (djk[0][end-1],djk[1][end-1])
