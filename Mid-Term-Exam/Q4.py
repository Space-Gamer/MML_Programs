# (a) Write a python program to output a set of orthonormal vectors whose span is the same as
# that of the following vectors.
# [1, 2, 3] ; [4, 5, 6] ; [7, 8, 9]
# (b) Generate 5 random vectors in R4 and apply the above program to get a set of orthonormal
# vectors.

import numpy as np
import time


def vector_norm(x, p=2):
    x_norm = 0
    for x_i in x:
        x_norm += (abs(x_i) ** p)
    x_norm = x_norm ** (1 / p)
    return x_norm


def gram_schmidt_process(v1, v2, v3, v4=None):
    if v4 is None:
        q = np.zeros((3, 3))
    else:
        q = np.zeros((4, 4))

    # Gram-Schmidt process
    q[0] = v1 / vector_norm(v1)
    q[1] = v2 - np.dot(v2, q[0]) * q[0]
    q[1] = q[1] / vector_norm(q[1])
    q[2] = v3 - np.dot(v3, q[0]) * q[0] - np.dot(v3, q[1]) * q[1]
    q[2] = q[2] / vector_norm(q[2])

    if v4 is not None:
        q[3] = v4 - np.dot(v4, q[0]) * q[0] - np.dot(v4, q[1]) * q[1] - np.dot(v4, q[2]) * q[2]
        q[3] = q[3] / vector_norm(q[3])

    return q


def main_a():  # For sub-question a
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    v3 = np.array([7, 8, 9])

    q = gram_schmidt_process(v1, v2, v3)

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

        q = gram_schmidt_process(v1, v2, v3, v4)

        print("Orthonormal vectors:")
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
