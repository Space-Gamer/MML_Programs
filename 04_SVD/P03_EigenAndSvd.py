#
# Author: Nagabhushan S N
# Last Modified: 25/02/2022
import math

import numpy
from matplotlib import pyplot


def demo1():
    matrix = numpy.array([[4, 1], [1, 4]])
    evals, evecs = numpy.linalg.eig(matrix)
    print(evals)
    print(evecs)
    return


def demo2():
    matrix = numpy.array([[4, 4], [0, 4]])
    u, s, vh = numpy.linalg.svd(matrix)
    print(u)
    print(s)
    print(vh)

    recon_matrix = numpy.matmul(u, numpy.matmul(numpy.diag(s), vh))
    print(recon_matrix)
    return


def normalize(x: numpy.ndarray, p: int):
    x_norm = numpy.linalg.norm(x, ord=p, axis=0, keepdims=True)
    normalized_x = x / x_norm
    return normalized_x


def unit_ball(p: int, num_points=100):
    theta = numpy.linspace(0, 2*math.pi, num_points)
    x = numpy.array([numpy.cos(theta), numpy.sin(theta)])
    x = normalize(x, p)
    return x


def plot_polygon(vertices: numpy.ndarray):
    vertices = numpy.concatenate([vertices, vertices[:, :1]], axis=1)
    pyplot.plot(vertices[0], vertices[1])
    return


def demo3():
    """
    Visualizing SVD matrix transformation as rotate-scale-rotate
    :return:
    """
    matrix = numpy.array([[4, 1], [1, 4]])
    u, s, vh = numpy.linalg.svd(matrix)
    x = unit_ball(p=2, num_points=1000)
    y = numpy.matmul(matrix, x)
    max_lim = max(numpy.abs(x).max(), numpy.abs(y).max()) + 2

    pyplot.figure(1, figsize=(8, 8))
    plot_polygon(x)
    plot_polygon(y)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    # pyplot.show()

    # Ax = u * sigma * vh * x

    vx = numpy.matmul(vh, x)
    svx = numpy.matmul(numpy.diag(s), vx)
    usvx = numpy.matmul(u, svx)
    pyplot.figure(2, figsize=(8, 8))
    pyplot.subplot(221)
    plot_polygon(x)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(222)
    plot_polygon(vx)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(223)
    plot_polygon(svx)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(224)
    plot_polygon(usvx)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.show()
    return


def main():
    # demo1()
    # demo2()
    demo3()
    return


if __name__ == '__main__':
    main()
