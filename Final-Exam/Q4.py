# . In the class, we interpreted SVD visually. Specifically, we found through SVD that multiplying
# any vector by a matrix is a sequence of 3 operations. Rotating the vector using V
# H, scaling the
# vector along different axes using Σ, and finally rotating the vector again using U. Similarly, write
# a python program to interpret eigen value decomposition (EVD) for the following two matrices.
# Do you observe any limitation of EVD that is not present in SVD.

# Matrix 1
# [[4 1]
#  [1 4]]

# Matrix 2
# [[4 4]
#  [0 4]]

import numpy as np
import matplotlib.pyplot as plt


def normalize(x: np.ndarray, p: int):
    x_norm = np.linalg.norm(x, ord=p, axis=0, keepdims=True)
    normalized_x = x / x_norm
    return normalized_x


def unit_ball(p: int, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = np.array([np.cos(theta), np.sin(theta)])
    x = normalize(x, p)
    return x


def plot_polygon(vertices: np.ndarray):
    vertices = np.concatenate([vertices, vertices[:, :1]], axis=1)
    plt.plot(vertices[0], vertices[1])
    return


def visualize_evd(matrix):
    evals, evecs = np.linalg.eig(matrix)
    print("Evals: ", evals)
    print("Evecs: ", evecs)
    Lambda = np.diag(evals)

    S_inv = np.linalg.inv(evecs)

    if np.linalg.matrix_rank(S_inv) < 2:
        print("Matrix is linearly dependent and diagonalizing is not possible")
        return
    else:
        print("Matrix is diagonalizable")

    x = unit_ball(p=1, num_points=1000)
    y = np.matmul(matrix, x)
    max_lim = max(np.abs(x).max(), np.abs(y).max()) + 2

    plt.figure(1, figsize=(8, 8))
    plot_polygon(x)
    plot_polygon(y)
    plt.xlim([-max_lim, max_lim])
    plt.ylim([-max_lim, max_lim])
    plt.suptitle(f"For matrix {matrix}")
    plt.title('Final transformation')
    plt.legend(['Original', 'Transformed'])

    p = np.matmul(S_inv, x)
    q = np.matmul(Lambda, p)
    r = np.matmul(evecs, q)

    plt.figure(2, figsize=(8, 8))
    plt.subplot(221)
    plot_polygon(x)
    # plot_polygon(p)
    plt.xlim([-max_lim, max_lim])
    plt.ylim([-max_lim, max_lim])
    plt.title('Unit Ball')

    plt.subplot(222)
    plot_polygon(p)
    # plot_polygon(q)
    plt.xlim([-max_lim, max_lim])
    plt.ylim([-max_lim, max_lim])
    plt.title('Transformation due to $\mathregular{S^{-1}}$')

    plt.subplot(223)
    plot_polygon(q)
    # plot_polygon(r)
    plt.xlim([-max_lim, max_lim])
    plt.ylim([-max_lim, max_lim])
    plt.title('Transformation due to \u039B')

    plt.subplot(224)
    # plot_polygon(x)
    plot_polygon(r)
    plt.xlim([-max_lim, max_lim])
    plt.ylim([-max_lim, max_lim])
    plt.title('Transformation due to S')

    plt.suptitle(f"For matrix {matrix}")
    plt.show()


def main():
    matrix1 = np.array([[4, 1], [1, 4]])
    print("\nMatrix 1: ", matrix1, "\n")
    visualize_evd(matrix1)

    matrix2 = np.array([[4, 4], [0, 4]])
    print("\nMatrix 2: ", matrix2, "\n")
    visualize_evd(matrix2)


if __name__ == '__main__':
    main()
