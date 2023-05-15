import numpy as np
from scipy.spatial import distance
from numpy import linalg as LA
from scipy import stats
from scipy.spatial import distance
import sys
import pandas as pd



class Topsis():
    #def __init__(self):




    def normalizacao_topsis(self, X):
        if X.ndim > 2:
            X_norm1 = X / LA.norm(X, axis=(0, 1))

        else:
            X_norm1 = X / LA.norm(X, axis=0)

        return X_norm1


    def ponderar(self, X, v_p):
        # A multiplicação dos pesos é feita para cada critério igualmente em todos os períodos de tempo
        tensor_ponderado = np.array([(np.multiply(i, v_p)).tolist() for i in X])

        return tensor_ponderado


    #O ideal e nadir abaixo é para o Topsis normal e para o algoritmo do artigo intervalar
    def ideal_nadir(self, X, crit_maxmin):

        a_ideal = []
        a_nadir = []

        dim = X.shape


        for j in range(dim[-1]):

            if X.ndim > 2:
                a_ideal.append(np.amax(X[:,:,j]) if crit_maxmin[j] else np.amin(X[:,:,j]))
                a_nadir.append(np.amin(X[:,:,j]) if crit_maxmin[j] else np.amax(X[:,:,j]))

            else:
                a_ideal.append(np.amax(X[:, j]) if crit_maxmin[j] else np.amin(X[:, j]))
                a_nadir.append(np.amin(X[:, j]) if crit_maxmin[j] else np.amax(X[:, j]))

        return np.array(a_ideal), np.array(a_nadir)



    # O ideal e nadir abaixo é para o algoritmo Topisis que propus
    def ideal_nadir_novo_algoritmo(self, X, crit_maxmin):

        a_ideal = []
        a_nadir = []

        for i, matriz in enumerate(X):
            a_ideal.append([np.amax(m, axis=0) if crit_maxmin[j] else np.amin(m, axis=0) for j, m in enumerate(matriz.T)])
            a_nadir.append([np.amin(m, axis=0) if crit_maxmin[j] else np.amax(m, axis=0) for j, m in enumerate(matriz.T)])

        return np.array(a_ideal), np.array(a_nadir)

    # O cálculo da distância abaixo é para o Topsis normal e para o algoritmo do artigo intervalar
    def distancia_id_nad(self, X, ideal, nadir, crit_max):

        dim = X.shape
        d_ideal = []
        d_nadir = []

        X_concat = []
        if X.ndim > 2:
            for j in range(dim[2]):
                if crit_max[j]:
                    X_concat.append(X[0,:,j])
                else:
                    X_concat.append(X[1, :, j])

            X_concat = np.array(X_concat).T
        else:
            X_concat = X

        for alternativa in X_concat:
            d_ideal.append(distance.euclidean(alternativa, ideal))
            d_nadir.append(distance.euclidean(alternativa, nadir))

        d_ideal = np.array(d_ideal)
        d_nadir = np.array(d_nadir)

        return d_ideal, d_nadir



    # O cálculo da distância abaixo é para o Topsis normal e para o algoritmo Topisis que propus
    def distancia_id_nad_novo_algoritmo(self, X, ideal, nadir, crit_max):

        # Junto a dimensão do axis 0 do meu tensor, transformo ele em uma matriz.
        # Assim o que consigo é que todos os critérios de uma alternativa estejam juntos em uma linha

        X_concat = np.concatenate(([X[i] for i in range(np.size(X, 0))]), axis=1)

        ideal_concat = np.reshape(ideal, ideal.shape[0] * ideal.shape[1])
        nadir_concat = np.reshape(nadir, nadir.shape[0] * nadir.shape[1])

        d_ideal = []
        d_nadir = []
        for i in X_concat:
            d_ideal.append(distance.euclidean(i, ideal_concat))
            d_nadir.append(distance.euclidean(i, nadir_concat))

        d_ideal = np.array(d_ideal)
        d_nadir = np.array(d_nadir)

        return d_ideal, d_nadir





    def calc_e_ordenamento_v(self, d_ideal, d_nadir, alt):

        v = d_nadir / (d_ideal + d_nadir)


        index = [i for i in range(len(v))]
        v_ordenado = sorted(zip(v, alt, index), key=lambda x: x[0], reverse=True)

        return v_ordenado





    def run(self):
        pass

    def ponderar_terceira_dimensao(self, X, vp_t):
        X_ponderada = []
        for i, matx_atri in enumerate(X):
            X_ponderada.append(matx_atri * vp_t[i])

        return np.array(X_ponderada)

    def trans_espaco_atributos(self, X):
        # Crio um vetor que determina se o atributo é de max ou min independente de se o critéiro é de max ou min.
        # Se o atributo depende do critério coloco False, se for sempre de min coloco 1, se for sempre de de max coloco 2
        atri_minmax = []

        # Normalizo os atributos dimensionais
        mean = np.mean(X, axis=0)

        # print('media')
        # print(mean)
        mean = self.normalizacao_topsis(mean)
        atri_minmax.append(False)

        # std = np.std(X, axis=0)
        # std = self.normalizacao_topsis(std)
        # atri_minmax.append(1)

        # var = np.var(X, axis=0)
        # var = self.normalizacao_topsis(var)
        # atri_minmax.append(2)

        cv = stats.variation(X, axis=0)
        atri_minmax.append(1)
        # print('cv')

        # O coeficiente angular (ca) depende do critério pois se o critério é de custo, é de mínimo, se é de benefício, é de máximo
        ca = []
        for i in X.T:
            aux = []
            for j in i:
                slope, intercept, r_value, p_value, std_err = stats.linregress((np.arange(len(j)), j))
                # print('slo', slope)
                aux.append(slope)
            ca.append(aux)

        # print('ca')
        ca = np.array(ca).T
        df = pd.DataFrame(ca)
        df.columns = pd.RangeIndex(1, len(df.columns) + 1)
        # print(df.to_latex())

        ca = self.normalizacao_topsis(ca)

        atri_minmax.append(False)

        # print('teste', X[-1])
        valor_temporal_mais_atual = X[-1]
        valor_temporal_mais_atual = self.normalizacao_topsis(valor_temporal_mais_atual)
        atri_minmax.append(False)

        # Verificar se para cada atributo eu normalizei e se coloquei se ele é atributo de min, max ou se depende do critério
        tensor = np.array([mean, cv, ca, valor_temporal_mais_atual])
        # tensor = np.array([mean, cv, ca])

        # Vetor de pesos terceira dimensão (temporal) vp_t
        # vp_t = [0.10, 0.10, 0.10, 0.70]
        # vp_t = [0.5, 0.25, 0.25]

        return tensor, atri_minmax
        # vetor de pesos
        #tensor = np.array([
        #    [[1,2,3],
        #     [4,5,6]],

        #    [[7,8,9],
        #     [1,2,3]]
        #])

        #crit_max = [True] * 3
        # nome das alternativas
        #alt = ['  a%d' % (d + 1) for d in range(len(tensor[0]))]





        # Normalizo o tensor temporal, tensor_n signigica o tensor normalizado




        # Mudo a terceira dimensão de série temporal para atributo sem normalizar, tensor_ea é o tensor espaço de atributo
        esp_atributo_tensor, atri_minmax, vp_t = self.trans_espaco_atributos(tensor)




        # O tensor do espaço de atributos eu pondero tanto os critérios como as séries temporais
        #esp_atributo_ponderado_terceira_dimensão = self.ponderar_terceira_dimensao(esp_atributo_tensor, vp_t)
        #esp_atributo_ponderado = self.ponderar(esp_atributo_ponderado_terceira_dimensão, vp)




        #ideal_esp_atributo_ponderado, nadir_esp_atributo_ponderado = self.ideal_nadir(esp_atributo_ponderado, crit_max, atri_minmax)



        #esp_atributo_d_ideal, esp_atributo_d_nadir = self.distancia_id_nad(esp_atributo_ponderado, ideal_esp_atributo_ponderado, nadir_esp_atributo_ponderado)



        #esp_atributo_v_ordenado = self.calc_e_ordenamento_v(esp_atributo_d_ideal, esp_atributo_d_nadir, alt)

        #print('temporal')
        #print(temporal_v_ordenado)
        #print('atributo')
        #print(esp_atributo_v_ordenado)

        #return temporal_v_ordenado, esp_atributo_v_ordenado





# A partir daqui é código antigo

#abaixo o tensor usado no artigo an algorithmic method to extend topsis for decision-making problems with interval data
    tensor = np.array([
        [[500.37, 2696995, 26364, 965.97],
         [873.7, 1027546, 3791, 2285.03],
         [95.93, 1145235, 22964, 207.98],
         [848.07, 390902, 492, 63.32],
         [58.69, 144906, 18053, 176.58],
         [464.39, 408163, 40539, 4654.71],
         [155.29, 335070, 33797, 560.26],
         [1752.31, 700842, 1437, 58.89],
         [244.34, 641680, 11418, 1070.81],
         [730.27, 453170, 2719, 375.07],
         [454.75, 309670, 2016, 936.62],
         [303.58, 286149, 14918, 1203.79],
         [658.81, 321435, 6616, 200.36],
         [420.18, 618105, 24425, 2781.24],
         [144.68, 119948, 1494, 282.73], ],

        [[961.37, 3126798, 38254, 6957.33],
         [1775.5, 1061260, 50308, 3174],
         [196.39, 1213541, 26846, 510.93],
         [1752.66, 395241, 1213, 92.3],
         [120.47, 165818, 18061, 370.81],
         [955.61, 416416, 48643, 5882.53],
         [342.89, 410427, 44933, 2506.67],
         [3629.54, 768593, 1519, 86.86],
         [495.78, 696338, 24108, 2283.08],
         [1417.11, 481943, 2955, 559.85],
         [931.24, 342598, 2617, 1468.45],
         [630.01, 317186, 27070, 4335.24],
         [1345.58, 347848, 8045, 399.8],
         [860.79, 835839, 40457, 4555.42],
         [292.15, 120208, 1749, 471.22], ]

    ])

    tensor = np.array([
        [[2, 1],
         [1, 5]],

        [[3, 2],
         [9, 4]],

        [[2, 3],
         [3, 3]]
    ])






        # Na função topisis para tensor 1, calculo as distâncias em cada período de tempo e depois a média temporal dessas distâncias.
        # Posteriormente aplico os pesos para cada critério
        #def topsis_para_tensor_1(self, X_normalizada, vetor_pesos, alt):
            # A multiplicação dos pesos é feita para cada critério igualmente em todos os períodos de tempo
            #X_normalizada_ponderada = np.multiply(X_normalizada, vetor_pesos)

            # Abaixo pego o maior e o menor valor das colunas incluida a terceira dimensão do tensor (que seria a série temporal)
            #a_mais = np.amax(X_normalizada_ponderada, axis=(0, 1))
            #a_menos = np.amin(X_normalizada_ponderada, axis=(0, 1))

            # Calculo a distância euclidiana para cada alternativa e para cada período
            #dist_max = []
            #dist_min = []
            #for j in X_normalizada_ponderada:
                #dist_max.append([distance.euclidean(a_mais, i) for i in j])
                #dist_min.append([distance.euclidean(a_menos, i) for i in j])

            # Calculo a média das distâncias dos períodos
            #dist_max_media_temporal = np.mean(dist_max, axis=0)
            #dist_min_media_temporal = np.mean(dist_min, axis=0)

            # Aplico o peso para cada critério
            # X_normalizada_ponderada = np.multiply(X_normalizada, vetor_pesos)

            #v = dist_min_media_temporal / (dist_max_media_temporal + dist_min_media_temporal)

            #index = [i for i in range(len(v))]
            #v_ordenado = sorted(zip(v, alt, index), key=lambda x: x[0], reverse=True)

            #return v_ordenado


    def topsis_para_matriz(self, X_normalizada, vetor_pesos, alt):
        # print('x normalizada', X_normalizada)

        # A função multiply multiplica cada coluna da matriz X_nomalizada pelo correspondente elemento do vetor de pesos
        X_normalizada_ponderada = np.multiply(X_normalizada, vetor_pesos)

        a_mais = np.array(np.amax(X_normalizada_ponderada, axis=0))
        a_menos = np.array(np.amin(X_normalizada_ponderada, axis=0))


        d_mais = np.sqrt(np.sum(np.power((a_mais - X_normalizada_ponderada), 2), axis=1))

        d_menos = np.sqrt(np.sum(np.power((a_menos - X_normalizada_ponderada), 2), axis=1))


        v = d_menos / (d_mais + d_menos)
        index = [i for i in range(len(v))]
        v_ordenado = sorted(zip(v, alt, index), key=lambda x: x[0], reverse=True)

        return v_ordenado




    # Na função topisis para tensor 1, calculo as distâncias em cada período de tempo e depois a média temporal dessas distâncias.
    #Posteriormente aplico os pesos para cada critério
    def topsis_para_tensor_1(self, X_normalizada_ponderada, vetor_pesos, vetor_pesos_terceira_dim, alt):

        # A multiplicação dos pesos é feita para cada critério igualmente em todos os períodos de tempo
        tensor_atr_ponderado = [(np.multiply(i, vetor_pesos)).tolist() for i in X]


        # Abaixo pego o maior e o menor valor das colunas incluida a terceira dimensão do tensor (que seria a série temporal)
        a_mais = np.amax(X_normalizada_ponderada, axis=(1))
        a_menos = np.amin(X_normalizada_ponderada, axis=(1))

        # Calculo a distância euclidiana para cada alternativa e para cada período
        dist_max = []
        dist_min = []
        for k, j in enumerate(X_normalizada_ponderada):
            dist_max.append([distance.euclidean(a_mais[k], i) for i in j])
            dist_min.append([distance.euclidean(a_menos[k], i) for i in j])


        # Mutiplico o resultado pelo peso determinado para a terceira dimensão, assim faço a primeira agregação
        dist_max_media_temporal = np.mean(dist_max, axis=0)
        dist_min_media_temporal = np.mean(dist_min, axis=0)




        v = dist_min_media_temporal / (dist_max_media_temporal + dist_min_media_temporal)

        index = [i for i in range(len(v))]
        v_ordenado = sorted(zip(v, alt, index), key=lambda x: x[0], reverse=True)

        return v_ordenado
