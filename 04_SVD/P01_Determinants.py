# Determinants
# Author: Nagabhushan S N
# Last Modified: 23/02/2022

import numpy


def compute_determinant(matrix: numpy.ndarray):
    m, n = matrix.shape
    assert m == n, "matrix should be square"

    if m == 1:
        determinant = matrix[0,0]
    else:
        determinant = 0
        for i in range(m):
            sub_matrix = numpy.concatenate([matrix[1:, 0:i], matrix[1:, i+1:]], axis=1)
            determinant += ((-1)**i) * matrix[0, i] * compute_determinant(sub_matrix)
    return determinant


def demo1():
    matrix = numpy.random.random(size=(3, 3))
    print(compute_determinant(matrix))
    print(numpy.linalg.det(matrix))
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    main()
