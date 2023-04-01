# (a) Sample 1000 vectors in R5
# randomly. Write a python script to show that, for each sampled
# vector x, the following holds.
# ∥x∥5 ≤ ∥x∥4 ≤ ∥x∥3 ≤ ∥x∥2 ≤ ∥x∥1
# (b) Check if the following holds.
# ∥x∥6 ≤ ∥x∥5

import numpy as np
import time


def vector_norm(x, p=2):
    x_norm = 0
    for x_i in x:
        x_norm += (abs(x_i) ** p)
    x_norm = x_norm ** (1 / p)
    return x_norm


def main_a():  # For sub-question a

    num_simulations = 1000
    dimension = 5

    for _ in range(num_simulations):

        x = np.random.rand(dimension)
        x_norm_5 = vector_norm(x, p=5)
        x_norm_4 = vector_norm(x, p=4)
        x_norm_3 = vector_norm(x, p=3)
        x_norm_2 = vector_norm(x, p=2)
        x_norm_1 = vector_norm(x, p=1)

        if x_norm_5 > x_norm_4 > x_norm_3 > x_norm_2 > x_norm_1:
            print(f'Cauchy Schwartz not satisfied for x = {x}')
            return

    print('Cauchy Schwartz satisfied for all simulations')
    return


def main_b():  # For sub-question b

    num_simulations = 1000
    dimension = 5

    for _ in range(num_simulations):

        x = np.random.rand(dimension)
        x_norm_6 = vector_norm(x, p=6)
        x_norm_5 = vector_norm(x, p=5)

        if x_norm_6 > x_norm_5:
            print(f'Cauchy Schwartz not satisfied for x = {x}')
            return

    print('Cauchy Schwartz satisfied for all simulations')
    print('∥x∥6 ≤ ∥x∥5 is true for all x in R5')
    return


if __name__ == '__main__':
    print('Sub-question a:')
    main_a()
    time.sleep(2)
    print('\n\nSub-question b:')
    main_b()

# Results: Cauchy Schwartz satisfied for all simulations. ∥x∥6 ≤ ∥x∥5 is true for all x in R5
#          Generally, ||x||p <= ||x||q for all p > q according to Cauchy Schwartz inequality.
