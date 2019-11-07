import csv
import math

def copiar_lista_instancias(lista):
    nova_lista = []
    for instancia in lista:
        nova_instancia = {}
        nova_instancia["Id"] = instancia["Id"]
        nova_instancia["SepalLengthCm"] = instancia["SepalLengthCm"]
        nova_instancia["SepalWidthCm"] = instancia["SepalWidthCm"]
        nova_instancia["PetalLengthCm"] = instancia["PetalLengthCm"]
        nova_instancia["PetalWidthCm"] = instancia["PetalWidthCm"]
        nova_instancia["Species"] = instancia["Species"]
        nova_lista.append(nova_instancia)
    return nova_lista

def imprimir_instancia(instancia):
    print(instancia["Id"], "-", instancia["SepalLengthCm"], "-", instancia["SepalWidthCm"], "-", instancia["PetalLengthCm"], "-", instancia["PetalWidthCm"], "-", instancia["Species"])

class KNN:

    tipos = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    def __init__(self, k):
        self.k = k
        self.lista_instancias = self.ler_arquivo("iris.csv")
        lista_instancias_novas = self.ler_arquivo("novas-instancias.csv")
        self.inicializar_matriz_confusao()

        for nova_instancia in lista_instancias_novas:
            self.lista_distancias = self.calcular_distancias(nova_instancia)
            self.cpy_lista_instancias = copiar_lista_instancias(self.lista_instancias)
            self.classificar(nova_instancia)
        self.imprimir_matriz_confusao()

    def ler_arquivo(self, caminho):
        arquivo  = open(caminho)
        lista_i = csv.DictReader(arquivo)
        lista_instancias = []
        for i in lista_i:
            instancia = {}
            instancia["Id"] = int(i["Id"])
            instancia["SepalLengthCm"] = float(i["SepalLengthCm"])
            instancia["SepalWidthCm"] = float(i["SepalWidthCm"])
            instancia["PetalLengthCm"] = float(i["PetalLengthCm"])
            instancia["PetalWidthCm"] = float(i["PetalWidthCm"])
            instancia["Species"] = str(i["Species"])
            lista_instancias.append(instancia)
        return lista_instancias

    def inicializar_matriz_confusao(self):
        self.matriz_confusao = [[0, 0, 0],\
                                [0, 0, 0],\
                                [0, 0, 0]]

    def calcular_distancias(self, nova_instancia):
        lista_distancias = []
        for instancia in self.lista_instancias:
            dx = nova_instancia["SepalLengthCm"] - instancia["SepalLengthCm"]
            dy = nova_instancia["SepalWidthCm"] - instancia["SepalWidthCm"]
            dz = nova_instancia["PetalLengthCm"] - instancia["PetalLengthCm"]
            dw = nova_instancia["PetalWidthCm"] - instancia["PetalWidthCm"]
            distancia = round((dx**2 + dy**2 + dz**2 + dw**2)**(1/2), 2)
            lista_distancias.append(distancia)
        return lista_distancias

    def classificar(self, nova_instancia):
        tipos = []
        quantidades = []
        for i in range(self.k):
            vizinho = self.instancia_mais_proxima()
            if vizinho["Species"] in tipos:
                pos = tipos.index(vizinho["Species"])
                quantidades[pos] += 1
            else:
                tipos.append(vizinho["Species"])
                quantidades.append(1)
        pos_maior = quantidades.index(max(quantidades))
        tipo_verdadeiro = nova_instancia["Species"]
        tipo_resultado = tipos[pos_maior]
        self.contabilizar_na_matriz(tipo_verdadeiro, tipo_resultado)

    def instancia_mais_proxima(self):
        pos = self.pos_menor_distancia()
        instancia = self.cpy_lista_instancias.pop(pos)
        return instancia

    def pos_menor_distancia(self):
        menor = math.inf
        pos_menor = -1
        for dist in self.lista_distancias:
            if dist < menor:
                pos_menor = self.lista_distancias.index(dist)
                menor = dist
        self.lista_distancias.pop(pos_menor)
        return pos_menor

    def contabilizar_na_matriz(self, tipo_verdadeiro, tipo_resultado):
        i = self.converter_para_indice(tipo_verdadeiro)
        j = self.converter_para_indice(tipo_resultado)
        self.matriz_confusao[i][j] += 1

    def converter_para_indice(self, tipo):
        if tipo == self.tipos[0]:
            return 0
        elif tipo == self.tipos[1]:
            return 1
        elif tipo == self.tipos[2]:
            return 2
        else:
            return "ERRO"

    def imprimir_matriz_confusao(self):
        print("Matriz de confusao (k = "+str(self.k)+")")
        for linha in range(3):
            print(self.tipos[linha].ljust(30, " "), self.matriz_confusao[linha])

#-------------------------------------------------------------------------------

knn = KNN(1)
print()
knn = KNN(3)
print()
knn = KNN(5)
print()
knn = KNN(7)
