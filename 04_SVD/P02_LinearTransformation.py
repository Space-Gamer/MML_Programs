# Matrix multiplication as a Linear Transform and Eigen values
# Author: Nagabhushan S N
# Last Modified: 23/02/2022
import math

import numpy
from matplotlib import pyplot, animation


def display_transformation(matrix: numpy.ndarray, vector: numpy.ndarray, x_min=-6, x_max=6, y_min=-6, y_max=6):
    if vector.ndim == 1:
        vector = vector[:, None]
    transformed_vector = numpy.matmul(matrix, vector)
    pyplot.quiver(vector[0, 0], vector[1, 0], angles='xy', scale_units='xy', scale=1, width=0.003, color='red')
    pyplot.quiver(transformed_vector[0, 0], transformed_vector[1, 0], angles='xy', scale_units='xy', scale=1, width=0.003, color='blue')
    pyplot.xlim([x_min, x_max])
    pyplot.ylim([y_min, y_max])
    pyplot.grid()
    pyplot.show()
    return


def normalize(x: numpy.ndarray, p: int):
    x_norm = numpy.linalg.norm(x, ord=p, axis=0, keepdims=True)
    normalized_x = x / x_norm
    return normalized_x


def unitball_transformation(matrix: numpy.ndarray, x_min=-6, x_max=6, y_min=-6, y_max=6):
    pyplot.xlim([x_min, x_max])
    pyplot.ylim([y_min, y_max])
    pyplot.grid()

    num_vectors = 10
    theta = numpy.linspace(0, 2*math.pi, num_vectors)
    x = numpy.array([numpy.cos(theta), numpy.sin(theta)])
    x = normalize(x, p=2)
    for i in range(num_vectors):
        vector = x[:, i:i+1]
        transformed_vector = numpy.matmul(matrix, vector)
        pyplot.quiver(vector[0, 0], vector[1, 0], angles='xy', scale_units='xy', scale=1, width=0.003, color='red')
        pyplot.quiver(transformed_vector[0, 0], transformed_vector[1, 0], angles='xy', scale_units='xy', scale=1, width=0.003, color='blue')
    pyplot.show()
    return


def animate_transformation(matrix: numpy.ndarray, x_min=-6, x_max=6, y_min=-6, y_max=6):
    # https://stackoverflow.com/a/19338495/3337089
    def update_quiver(num, q, x, y):
        num = num % x.size
        q.set_UVC(x[num], y[num])
        return q

    num_vectors = 100
    theta = numpy.linspace(0, 2*math.pi, num_vectors)
    vectors = numpy.array([numpy.cos(theta), numpy.sin(theta)])
    vectors = normalize(vectors, p=2)
    transformed_vectors = numpy.matmul(matrix, vectors)

    fig = pyplot.figure(figsize=(8, 8))
    pyplot.xlim([x_min, x_max])
    pyplot.ylim([y_min, y_max])
    pyplot.grid()
    q1 = pyplot.quiver(0, 0, angles='xy', scale_units='xy', scale=1, width=0.003, color='red')
    q2 = pyplot.quiver(0, 0, angles='xy', scale_units='xy', scale=1, width=0.003, color='blue')
    anim1 = animation.FuncAnimation(fig, update_quiver, fargs=(q1, vectors[0], vectors[1]), interval=50, blit=False)
    anim2 = animation.FuncAnimation(fig, update_quiver, fargs=(q2, transformed_vectors[0], transformed_vectors[1]), interval=50, blit=False)
    pyplot.show()
    return


def demo1():
    matrix = numpy.array([[4, 1], [1, 4]])
    vector = numpy.array([1, 0])
    display_transformation(matrix, vector)
    return


def demo2():
    matrix = numpy.array([[4, 1], [1, 4]])
    unitball_transformation(matrix)
    return


def demo3():
    matrix = numpy.array([[2, 2], [0, 2]])
    animate_transformation(matrix, x_min=-3, x_max=3, y_min=-3, y_max=3)
    return


def main():
    # demo1()
    # demo2()
    demo3()
    return


if __name__ == '__main__':
    main()
