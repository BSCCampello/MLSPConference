import numpy as np
import matplotlib.pylab as plt
import sys
import json
import scipy
from scipy import signal
from numpy.linalg import inv
#from padasip.filters.base_filter import AdaptiveFilter


# https://matousc89.github.io/padasip/sources/filters/rls.html

class FilterRLS_beta():


    def __init__(self):
        pass



    def run(self, d, x, passo, N, lamb, gama, parametros):


        N = len(d)

        try:
            x = np.array(x)
            d = np.array(d)
        except:
            raise ValueError('Impossible to convert x or d to a numpy array')

        # create empty arrays
        S = gama * np.identity(parametros)
        y = np.zeros(N -(passo-1))
        e = np.zeros(N -(passo-1))
        epi = np.zeros(N -(passo-1))
        self.w_history = np.zeros((N-(passo-1), parametros))
        self.w = np.zeros(parametros)
        d = d[passo-1:len(d)]

        # adaptation loop
        # algoritmo pag 213(218) -  Adaptive Filtering
        # Algorithms and Practical Implementation

        for k in range(0, N -(passo-1)):

            self.w_history[k, :] = self.w

            y[k] = np.dot(self.w.T, x[k])
            e[k] = d[k] - np.dot(x[k].T, self.w)
            #print('xk',x[k])
            #print('dk',d[k])
            #print('predicao', np.dot(x[k].T, self.w))



            a = np.dot(np.transpose(x[k]), S)

            b = 1 / (lamb + np.dot(a, x[k]))

            g_n = np.dot(S, x[k]) * b

            c = (1 / lamb) * S


            aux = np.zeros((parametros, parametros))
            for aa in range(len(g_n)):
                for bb in range(len(x[k])):
                    aux[aa][bb] = g_n[aa] * x[k][bb]


            S = c - np.dot(aux, c)
            aux2 = g_n * e[k]
            self.w = self.w + aux2



        predicao = np.dot(self.w, x[-1])
        #print('x-1',x[-1])
        #print('predicao rls a passo %d: %.5f'%(passo, predicao))

        return y, e, self.w_history, predicao




"""
for k in range(0, N -(passo-1)):
    print('iteração', k)
    print('\n')

    print('x[k]:', x[k].tolist())
    print('\n')

    print('S[k-1]:')
    print(S)

    self.w_history[k, :] = self.w
    print('w[k-1]:', self.w.tolist())
    print('\n')

    e[k] = d[k] - np.dot(x[k].T, self.w)
    print('x.T*w[k-1]:', np.dot(x[k].T, self.w))
    print('\n')
    print('e[k]', e[k])
    print('\n')

    fi = np.dot(S, x[k])
    print('fi', fi)
    print('\n')

    numerador = np.dot(fi, fi.T)
    denominador = lamb + (np.dot(fi.T, x[k]))
    print('numerador:', numerador)
    print('\n')
    print('denominador:', denominador)
    print('\n')

    S = 1/lamb * (S - (numerador/denominador))

    print('S[k]')
    print(S)
    print('\n')

    dw = e[k] * (np.dot(S, x[k]))

    print('dw', dw)
    print('\n')

    self.w += dw

    print('w[k]', self.w)
    print('\n')

    y[k] = np.dot(self.w.T, x[k])
    epi[k] = d[k] - y[k]
    print('y[k]:', y[k].tolist())
    print('\n')
    print('d[k]:', d[k].tolist())
    print('\n')
    print('e_aposteriori[k]:', epi[k].tolist())
    print('\n'),
"""
