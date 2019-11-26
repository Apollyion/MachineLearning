# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:12:09 2019

@author: lucas
"""

import matplotlib.pyplot as plt
import numpy as np

font = {'family':'arial','color':'black','weight':'normal','size': 18}

#Parte 1

algoritmo = ['SVM', 'KNeighbors', 'SGDClassifier']
acuracy1 = [97,99,89]
xs = [i + 0.5 for i, _ in enumerate(algoritmo)]
margem_erro = [3, 1, 7]
plt.bar(xs, acuracy1, yerr= margem_erro)

plt.xlabel('Algoritmos utilizados', fontdict=font)
plt.ylabel('Acurácia (%)', fontdict=font)

# Tweak spacing to prevent clipping of ylabel
plt.title('PARTE 1', fontdict=font)
plt.xticks([i + 0.5 for i, _ in enumerate(algoritmo)], algoritmo)
plt.subplots_adjust(left=0.15)
plt.show()


#tabela pt 1

fig = plt.figure(dpi=130)
ax = fig.add_subplot(1,1,1)
table_data=[
    ["Execução", 1,2,3,4,5,6,7,8,9,10],
    ["SVC", 97,97,97,97,97,97,97,97,97,97],
    ["CLF", 89,94,91,91,90,91,87,91,94,92],
    ["KNeighbors", 99,99,99,99,99,99,99,99,99,99],
]
table = ax.table(cellText=table_data, loc='center')
table.set_fontsize(18)
table.scale(1,1)
plt.title('10 execuções(PARTE 1)\nAcurácia individual')
ax.axis('off')
plt.show()


#Parte 2

algoritmo = ['GaussianNB', 'KMeans', 'NuSVC', 'RandomForest']
acuracy1 = [92,84,88,95]
xs = [i + 0.5 for i, _ in enumerate(algoritmo)]
plt.bar(xs, acuracy1)

plt.xlabel('Algoritmos utilizados', fontdict=font)
plt.ylabel('Acurácia ', fontdict=font)


plt.title('PARTE 2', fontdict=font)
plt.xticks([i + 0.5 for i, _ in enumerate(algoritmo)], algoritmo)
plt.subplots_adjust(left=0.15)
plt.show()


#Gerando tabela pt2
fig = plt.figure(dpi=130)
ax = fig.add_subplot(1,1,1)
table_data=[
    ["Execução", 1,2],
    ["GaussianNB", 0.90,0.94],
    ["KMeans", 0.71,0.89],
    ["NuSVC", 0.80,0.91],
    ["RandomForest ", 0.90,0.94],
]
table = ax.table(cellText=table_data, loc='center')
table.set_fontsize(18)
table.scale(1,1)
plt.title('2 execuções(PARTE 2)\nAcurácia individual')
ax.axis('off')
plt.show()




