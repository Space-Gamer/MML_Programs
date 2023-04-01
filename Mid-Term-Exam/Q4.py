# (a) Write a python program to output a set of orthonormal vectors whose span is the same as
# that of the following vectors.
# [1, 2, 3] ; [4, 5, 6] ; [7, 8, 9]
# (b) Generate 5 random vectors in R4 and apply the above program to get a set of orthonormal
# vectors.
# Note: Perform a check for independence of given vectors before applying Gram-Schmidt.

import numpy as np
import time


def vector_norm(x, p=2):
    x_norm = 0
    for x_i in x:
        x_norm += (abs(x_i) ** p)
    x_norm = x_norm ** (1 / p)
    return x_norm


def gram_schmidt_process(vectors):
    m, n = vectors.shape

    q = np.zeros((m, n))

    for i in range(m):
        v = vectors[i]
        for j in range(i):
            v = v - np.dot(v, q[j]) * q[j]
        q[i] = v / vector_norm(v)

    return q


def independent_vectors(vectors):
    m, n = vectors.shape

    ind_vect = []

    for i in range(m):
        v = vectors[i]
        for j in range(i):
            v = v - np.dot(v, ind_vect[j]) * ind_vect[j]
        if vector_norm(v) > 1e-10:
            ind_vect.append(v / vector_norm(v))

    return np.array(ind_vect)


def main_a():  # For sub-question a
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    v3 = np.array([7, 8, 9])

    print("The only independent vectors are:")
    ind_vect = independent_vectors(np.array([v1, v2, v3]))
    print(ind_vect)

    print("\nOrthonormal vectors are:")
    q = gram_schmidt_process(ind_vect)

    print(q)
    return


def main_b():  # For sub-question b
    num_simulations = 5
    dimension = 4

    for _ in range(num_simulations):
        v1 = np.random.rand(dimension)
        v2 = np.random.rand(dimension)
        v3 = np.random.rand(dimension)
        v4 = np.random.rand(dimension)

        print("Randomly generated vectors:")
        print("v1 = ", v1)
        print("v2 = ", v2)
        print("v3 = ", v3)
        print("v4 = ", v4)

        print("\nIndependent vectors are:")
        ind_vect = independent_vectors(np.array([v1, v2, v3, v4]))
        print(ind_vect)

        q = gram_schmidt_process(ind_vect)

        print("\nOrthonormal vectors:")
        print(q)
        print()
    return


if __name__ == '__main__':
    print('Sub-question a:')
    main_a()
    time.sleep(2)
    print('\n\nSub-question b:')
    main_b()

# Results: The required orthonormal vectors were obtained.
