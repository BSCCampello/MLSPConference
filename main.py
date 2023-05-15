import numpy as np
#import sys
import math
import json
from lms_teste import LMS
from kendal_tau import Kendal
from rls import FilterRLS_beta
from promethee import Promethee
#from lms_normalizado import LMS_n
from numpy import sqrt
import matplotlib.pylab as plt
#from promethee_dominance_degrees import Promethee_do_de




# Parameters:

n_iterations = 1000 # Number of interations - Monte Carlo simulation - Times to repet the test
gerar_sinais = False # True if the signals have to be gerenerate or False if just repeat the alredy gernerate signal

vec_passos = [1, 2, 3, 4, 5]


# <editor-fold desc="parametros">
nome_da_pasta = 'data' # folder name from the code search the file with the input data to run.
n_a = 5 # Number of alternatives


# Signal
N = 31  # Number of sample, it means the size of the signal
variancia_ruido = 1 # White noise variance
n_alfa = 51 # Quantity of diferents alpha, it is a linear factor which multiply the noise to increase it

# Adaptative filter
filt_paramet = 2  # Quantity of adaptative filtering parameters

# RLS:
lamb = 0.9 # The forgetting factor of the RLS
gama = 0.1 # To initialize the RLS R-matrix

# LMS:
mu_lms = 0.3

# Parameters for the MCDA method

v_p = [1/4, 1/4, 1/4, 1/4] # Weight vector
crit_max = [True] * n_a
alt = ['a%d' % (d + 1) for d in range(n_a)] # This vector is just to name the variables as a_1...a_n

nome_arquivo = 'sinais_simulacao_num_alternativas_%d'%n_a   # File name: the file to run depends on the alternatives' name

# Promethee
funçao_de_pref = [('usual', 0), ('usual', 0),  ('usual', 0),  ('usual', 0)]
# </editor-fold>

# The strcture is that:

# The first for is to make monte carlo simulation
# The second for is to increase the white noise by an alpha factor
# The third for is to take all alternatives of a given criterion
# The fourth for is to take each alternative of the criterion and calculate the signal prediction
# The last for is to the SMAA method, in each iteration, we take a radom number in the interval calculated in the step before


PR_sum_tau = np.zeros(n_alfa)
PR_sum_tau_por_periodo = np.zeros((n_alfa, len(vec_passos)))

L_sum_tau = np.zeros(n_alfa)
PR2_sum_tau = np.zeros(n_alfa)
sum_toal_desvio_do_erro = np.zeros(n_alfa)


def gerar_sinal():

    m_d_1 = []
    m_d_2 = []
    m_d_3 = []
    m_d_4 = []
    m_r = []

    for iteracao in range(n_iterations):

        # Signals criterion 1
        t = np.arange(1, N + 1)
        a = np.random.uniform(2, 20, n_a)
        b = np.random.uniform(0.5, 0.8, n_a)
        d_1 = []
        for i in range(len(a)):
            d = a[i] + b[i] * t
            d_1.append((d).tolist())
        m_d_1.append(d_1)

        # Signals criterion 2
        n = np.arange(1, N + 1)
        a = np.random.uniform(2, 20, n_a)
        freq = np.random.uniform(0.7, 0.75, n_a) * math.pi
        d_2 = []
        for i in range(len(freq)):
            d = a[i] + np.sin(freq[i] * n)
            d_2.append((d).tolist())
        m_d_2.append(d_2)

        # Signals criterion 3

        d_3 = []
        a = np.random.uniform(2, 20, n_a)
        for i in range(n_a):
            d = a[i] + (-1) ** t
            d_3.append((d).tolist())

        m_d_3.append(d_3)


        # Signals criterion 4
        r_i = np.random.uniform(1, 20, n_a)
        d_4 = []
        for i in range(len(r_i)):
            d = (r_i[i]) + t ** 0.075
            d_4.append((d).tolist())
        m_d_4.append(d_4)

        # The white noise vector:
        r = np.random.normal(0, math.sqrt(variancia_ruido), N)
        m_r.append(r.tolist())


    fp = open("%s/%s" % (nome_da_pasta, nome_arquivo), "w")
    dict_dados = {
        'd_1': m_d_1,
        'd_2': m_d_2,
        'd_3': m_d_3,
        'd_4': m_d_4,
        'r': m_r,
    }

    fp.write(json.dumps(dict_dados) + "\n")



def calc_ord(matriz_decisao, funçao_de_pref, v_p, promethee = 0):
    ranking = []
    score = []
    # PROMETHEE ordering, promethee = 0 is the classic one, promethee = 1 is with dominance in the prob function as in the article and promethee = 2 is dominance as I proposed
    if promethee == 0:
        R_promethee = Promethee()
        v_ordenado, score = R_promethee.run(matriz_decisao, crit_max, funçao_de_pref, v_p, alt)

    elif promethee == 1:
        #promethee_novo = Promethee_do_de()
        #v_ordenado = promethee_novo.run(matriz_decisao, crit_max, v_p, True)
        pass
    else:
        #promethee_novo = Promethee_do_de()
        #v_ordenado = promethee_novo.run(matriz_decisao, crit_max, v_p, False)
        pass

    ranking = []
    for i in range(len(v_ordenado)):
        ranking.append(v_ordenado[i][2])


    return ranking, score




if gerar_sinais:
    gerar_sinal()



# <editor-fold desc="LerDadosSinais">
fp = open("%s/%s" % (nome_da_pasta, nome_arquivo), "r")
for instancia in fp.readlines():
    dict_dados = json.loads(instancia)
    d_1 = dict_dados['d_1']
    d_2 = dict_dados['d_2']
    d_3 = dict_dados['d_3']
    d_4 = dict_dados['d_4']
    m_r = np.array(dict_dados['r'])
# </editor-fold>



c_tau_final = 0
for interation in range(n_iterations):

    print('ite: ', interation)

    sinais = [d_1[interation], d_2[interation], d_3[interation], d_4[interation]]
    r = m_r[interation]

    PR_aux_mean_tau = []
    PR_aux_mean_tau_por_periodo = []
    L_aux_mean_tau = []
    PR2_aux_mean_tau = []
    aux_media_desvio_do_erro = []

    for SNR in range(n_alfa):

        R_matriz_score_por_periodo = np.zeros((len(vec_passos), n_a))
        R_matriz_ranking_por_periodo = np.zeros((len(vec_passos), n_a))
        L_matriz_ranking_por_periodo = np.zeros((len(vec_passos), n_a))
        Ideal_matriz_ranking_por_periodo = np.zeros((len(vec_passos), n_a))
        Current_matriz_ranking_por_periodo = np.zeros((len(vec_passos), n_a))



        for t, passo in enumerate(vec_passos):

            R_mat_pred = np.zeros((len(sinais), n_a))  # make a transpose matrix to facilitate the way to compute values
            R_mat_pred_ideal = np.zeros((len(sinais), n_a))
            L_mat_pred = np.zeros((len(sinais), n_a))
            m_classica = np.zeros((len(sinais), n_a))
            mat_media_desvio = np.zeros((len(sinais), n_a), dtype=bytearray)
            desvio_medio_total = 0

            for c, criterion in enumerate(sinais):

                for a, sig_alt in enumerate(criterion):

                    # To calculate the alpha value
                    # <editor-fold desc="adicionando_ruido">
                    power = (10 * math.log10(np.var(sig_alt)) - SNR) / 20  # Esta é a fórmula para calcular de quanto tem que ser o alfa para cada SNR
                    alfa = math.pow(10, power)

                    # To add a noise in the signal
                    d_completo = sig_alt[:len(sig_alt)-1] + alfa * r[:len(r)-1]


                    #O sinal para a predição é o d_completo menos o tamanho do passo
                    d = d_completo[:len(d_completo)-passo]


                    # </editor-fold>

                    # Para gerar a referância
                    # <editor-fold desc="To_generate_ref_vector">
                    x = [[0] * filt_paramet]
                    aux = np.concatenate((np.zeros(filt_paramet), d))
                    for k in range(filt_paramet, len(d) + filt_paramet):
                        x.append([aux[k - i] for i in range(0, filt_paramet)])
                    # </editor-fold>

                    # Passo de predição com RLS e NLMS

                    # Filtrando com RLS
                    Rf = FilterRLS_beta()
                    Ry, Re, Rw, Rpredicao = Rf.run(d, x, passo, N, lamb, gama, filt_paramet)

                    # Filtrando com LMS
                    f = LMS()
                    Ly, Le, Lw, Lpredicao = f.run(d, x, passo, N, mu_lms, filt_paramet)


                    # Calculando o desvio do erro
                    # <editor-fold desc="desvio_e">
                    re_2 = np.square(Re[3:])
                    sum_re_2 = np.sum(re_2)
                    var_erro = (1 / (len(re_2) - filt_paramet)) * sum_re_2
                    desvio_erro = sqrt(var_erro)
                    desvio_medio_total += np.mean(desvio_erro)
                    # </editor-fold>

                    # Guardo o valor de predição de cada alternativa em cada critério
                    R_mat_pred[c][a] = Rpredicao
                    L_mat_pred[c][a] = Lpredicao

                    # Guardo o valor ideal em cada alternativa em cada critério
                    valor_ideal = d_completo[-1]
                    R_mat_pred_ideal[c][a] = valor_ideal

                    # Guardo o que seria o valor current data em cada alternativa em cada critério
                    m_classica[c][a] = d_completo[-1-passo]

                    # Em uma matriz, eu guardo a média e o desvio
                    mat_media_desvio[c][a] = np.array([Rpredicao, desvio_erro])






            # IMPORTANTE: preciso trabalhar com as matrizes transpostas
            R_mat_pred = np.transpose(R_mat_pred)
            L_mat_pred = np.transpose(L_mat_pred)
            R_mat_pred_ideal = np.transpose(R_mat_pred_ideal)
            m_classica = np.transpose(m_classica)

            mat_media_desvio = np.transpose(np.array(mat_media_desvio))
            aux_media_desvio_do_erro.append(desvio_erro / (len(sinais) * n_a))

            # <editor-fold desc="Cálculo dos rankings">
            PR_ranking_1, PR_score_1 = calc_ord(R_mat_pred, funçao_de_pref, v_p)

            PR_ideal_ranking, Ideal_score = calc_ord(R_mat_pred_ideal, funçao_de_pref, v_p)

            L_ranking, L_score = calc_ord(L_mat_pred, funçao_de_pref, v_p)

            classico_ranking, classico_score = calc_ord(m_classica, funçao_de_pref, v_p)

            R_matriz_score_por_periodo[t] = PR_score_1
            R_matriz_ranking_por_periodo[t] = PR_ranking_1
            L_matriz_ranking_por_periodo[t] = L_score
            Ideal_matriz_ranking_por_periodo[t] = Ideal_score
            Current_matriz_ranking_por_periodo[t] = classico_score


        # IMPORTANTE: preciso trabalhar com as matrizes transpostas
        R_matriz_score_por_periodo = np.transpose(R_matriz_score_por_periodo)
        R_matriz_ranking_por_periodo = np.transpose(R_matriz_ranking_por_periodo)
        L_matriz_ranking_por_periodo = np.transpose(L_matriz_ranking_por_periodo)
        Ideal_matriz_ranking_por_periodo = np.transpose(Ideal_matriz_ranking_por_periodo)
        Current_matriz_ranking_por_periodo = np.transpose(Current_matriz_ranking_por_periodo)

        funçao_de_pref_2 = [('usual', 0)] * len(R_matriz_score_por_periodo)
        v_peso = [(1 / len(R_matriz_score_por_periodo))] * len(R_matriz_score_por_periodo)


        PR_ranking, PR_score = calc_ord(R_matriz_score_por_periodo, funçao_de_pref_2, v_peso)
        L_ranking, L_score = calc_ord(L_matriz_ranking_por_periodo, funçao_de_pref_2, v_peso)
        PR_ideal_ranking, Ideal_score = calc_ord(Ideal_matriz_ranking_por_periodo, funçao_de_pref_2, v_peso)
        classico_ranking, classico_score = calc_ord(Current_matriz_ranking_por_periodo, funçao_de_pref_2, v_peso)



        # <editor-fold desc="calcular o tau para cada tipo de matriz">
        # To compare the ranking with the ideal decision matrix
        R_kendal = Kendal()
        R_tau = R_kendal.run(PR_ideal_ranking, PR_ranking)

        aux_por_periodo = []
        for p, passo in enumerate(vec_passos):
            R_kendal_por_periodo = Kendal()
            aux_por_periodo.append(R_kendal_por_periodo.run(PR_ranking, R_matriz_score_por_periodo[p]))



        L_kendal = Kendal()
        L_tau = L_kendal.run(PR_ideal_ranking, L_ranking)

        PR_aux_mean_tau.append(R_tau)
        L_aux_mean_tau.append(L_tau)
        PR_aux_mean_tau_por_periodo.append(aux_por_periodo)


        # comparar o ranking ideal com o classico ou current
        R2_kendal = Kendal()
        R2_tau = R2_kendal.run(PR_ideal_ranking, classico_ranking)
        PR2_aux_mean_tau.append(R2_tau)



    PR_sum_tau += PR_aux_mean_tau
    PR2_sum_tau += PR2_aux_mean_tau
    PR_sum_tau_por_periodo += (PR_aux_mean_tau_por_periodo)



    L_sum_tau += L_aux_mean_tau

R_tau_medio_predicao_ideal = PR_sum_tau/n_iterations
PR_aux_mean_tau_por_periodo = PR_sum_tau_por_periodo/n_iterations

L_tau_medio_predicao_ideal = L_sum_tau/n_iterations
tau_medio_current_ideal = PR2_sum_tau/n_iterations
#desvio_medio = sum_toal_desvio_do_erro/n_iterations



vec_SNR = np.arange(n_alfa)



# PLOTANDO RLS comparacao periodo a periodo
plt.figure(figsize=(30, 30))
linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1))] # define diferentes tipos de linha para cada elemento do vetor

for i in range(len(PR_aux_mean_tau_por_periodo[0])):
    x = range(len(PR_aux_mean_tau_por_periodo))
    y = [PR_aux_mean_tau_por_periodo[j][i] for j in range(len(PR_aux_mean_tau_por_periodo))]
    plt.plot(x, y, label=r'$\hat{\tau}_{ g_%d  \times  \hat{r}}$'%(i+1), linestyle=linestyles[i])

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
plt.title('Comparison', fontsize=40)
plt.xlim([-1, 51])
plt.show()





# PLOTANDO RLS
'''
min = np.min(R_tau_medio_predicao_ideal)
max = np.max(R_tau_medio_predicao_ideal)

plt.figure(figsize=(30, 30))
plt.plot(vec_SNR, R_tau_medio_predicao_ideal, label=r'$\hat{\tau}_{r^* \times  \hat{r}}$')
plt.plot(vec_SNR, tau_medio_current_ideal, linestyle='dashed',  label=r'$\tau^c_{r^* \times  r^c}$')
plt.xlabel(r"SNR", fontsize=35)
plt.ylabel(r"$\tau$" , fontsize=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-1, 51])
#plt.ylim([min, max])
plt.title('Prediction using RLS', fontsize=40)
plt.legend(fontsize=40)
plt.show()
'''

# PLOTANDO LMS
'''
min = np.min(L_tau_medio_predicao_ideal)
max = np.max(L_tau_medio_predicao_ideal)

plt.figure(figsize=(30, 30))
plt.plot(vec_SNR, L_tau_medio_predicao_ideal, label=r'$\hat{\tau}_{r^* \times  \hat{r}}$')
#plt.plot(vec_SNR, tau_medio_current_ideal , linestyle='dashed',  label=r'$\tau^c_{r^* \times  r^c}$')
plt.xlabel(r"SNR", fontsize=35)
plt.ylabel(r"$\tau$" , fontsize=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-1, 51])
#plt.ylim([min, max])
plt.title('Prediction using LMS', fontsize=40)
plt.legend(fontsize=40)
plt.show()
'''


data_r = {
        'tau_medio_Rpredicao_ideal': R_tau_medio_predicao_ideal.tolist(),
        'L_tau_medio_predicao_ideal': L_tau_medio_predicao_ideal.tolist(),
        'tau_medio_current_ideal': tau_medio_current_ideal.tolist(),

        }

#fp = open("%s/%s" % (nome_da_pasta, 'simulation_result_for_%d_alternatives'%n_a), "w")
#fp.write(json.dumps(data_r) + "\n")




