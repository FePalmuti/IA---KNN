import csv
import math
import sys

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
        self.calcular_metricas()

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
        print()
        for linha in range(3):
            print(self.tipos[linha].ljust(30, " "), self.matriz_confusao[linha])
        print()

    def calcular_metricas(self):
        tam = len(self.tipos)
        for pos in range(tam):
            print(self.tipos[pos])
            coeficientes = self.calcular_coeficientes(pos)
            a = coeficientes["a"]
            b = coeficientes["b"]
            c = coeficientes["c"]
            d = coeficientes["d"]
            print("    Acuracia =", self.acuracia(a, b, c, d))
            print("    Taxa de verdadeiros positivos =", self.vp(a, b, c, d))
            print("    Taxa de falsos positivos =", self.fp(a, b, c, d))
            print("    Taxa de verdadeiros negativos =", self.vn(a, b, c, d))
            print("    Taxa de falsos negativos =", self.fn(a, b, c, d))
            print("    Precisao =", self.precisao(a, b, c, d))
            print("    F-score =", self.f_score(a, b, c, d))
            print()

    def calcular_coeficientes(self, pos):
        tam = len(self.tipos)
        coeficientes = {"a":0, "b":0, "c":0, "d":0}
        coord = {"i":pos, "j":pos}
        for i in range(tam):
            for j in range(tam):
                if i==j:
                    if i==coord["i"] and j==coord["j"]:
                        coeficientes["a"] = self.matriz_confusao[i][j]
                    else:
                        coeficientes["d"] += self.matriz_confusao[i][j]
                else:
                    if i==coord["i"] and j!=coord["j"]:
                        coeficientes["b"] += self.matriz_confusao[i][j]
                    elif i!=coord["i"] and j==coord["j"]:
                        coeficientes["c"] += self.matriz_confusao[i][j]
        return coeficientes

    def acuracia(self, a, b, c, d):
        return round((a+b)/(a+b+c+d), 2)

    def vp(self, a, b, c, d):
        return round((d)/(c+d), 2)

    def fp(self, a, b, c, d):
        return round((b)/(a+b), 2)

    def vn(self, a, b, c, d):
        return round((a)/(a+b), 2)

    def fn(self, a, b, c, d):
        return round((c)/(c+d), 2)

    def precisao(self, a, b, c, d):
        return round((d)/(b+d), 2)

    def f_score(self, a, b, c, d):
        recall = self.vp(a, b, c, d)
        precisao = self.precisao(a, b, c, d)
        return round(2*(recall*precisao)/(recall+precisao), 2)


#-------------------------------------------------------------------------------

knn = KNN(int(sys.argv[1]))

#
