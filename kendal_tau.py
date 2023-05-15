import itertools

class Kendal():

    def __init__(self):
        pass

    def run(self, x, y):

        n = len(x)

        if len(y) != n:
            print('Os valores dos vetores não coincidem')


        pares_x = tuple(itertools.combinations(x, 2))
        pares_y = tuple(itertools.combinations(y, 2))



        compute = 0
        for i in range(len(pares_x)):
            if (pares_x[i][0] > pares_x[i][1] and pares_y[i][0] < pares_y[i][1]) or \
               (pares_x[i][0] < pares_x[i][1] and pares_y[i][0] > pares_y[i][1]):
                compute += 1

        kendall_tau = compute/( (n*(n-1))/2 )

        return kendall_tau
