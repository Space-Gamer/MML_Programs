# Inverses
# Author: Nagabhushan S N
# Last Modified: 25/02/2022

import numpy
from matplotlib import pyplot, animation
from P03_EigenAndSvd import unit_ball, plot_polygon


def demo1():
    matrix = numpy.array([[4, 1], [1, 4]])
    matrix_inv = numpy.linalg.inv(matrix)
    u1, s1, v1 = numpy.linalg.svd(matrix)
    u2, s2, v2 = numpy.linalg.svd(matrix_inv)
    print(u1, s1, v1, sep='\n')
    print(u2, s2, v2, sep='\n')

    x = unit_ball(p=1, num_points=1000)
    y = numpy.matmul(matrix, x)
    max_lim = max(numpy.abs(x).max(), numpy.abs(y).max()) + 2

    vx = numpy.matmul(v1, x)
    svx = numpy.matmul(numpy.diag(s1), vx)
    usvx = numpy.matmul(u1, svx)  # Ax

    y = usvx
    vy = numpy.matmul(v2, y)
    svy = numpy.matmul(numpy.diag(s2), vy)
    usvy = numpy.matmul(u2, svy)

    pyplot.figure(1, figsize=(8, 8))
    pyplot.subplot(241)
    plot_polygon(x)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(242)
    plot_polygon(vx)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(243)
    plot_polygon(svx)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(244)
    plot_polygon(usvx)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(245)
    plot_polygon(vy)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(246)
    plot_polygon(svy)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(247)
    plot_polygon(usvy)
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.show()
    return


def update_quiver(num, q, x, y):
    num = num % x.size
    q.set_UVC(x[num], y[num])
    return q


def demo2():
    """
    Transformation of an invertible matrix
    :return:
    """
    matrix = numpy.array([[4, 1], [1, 4]])
    matrix_inv = numpy.linalg.inv(matrix)

    x = unit_ball(p=2, num_points=100)
    y = numpy.matmul(matrix, x)
    z = numpy.matmul(matrix_inv, y)
    max_lim = max(numpy.abs(x).max(), numpy.abs(y).max())

    fig = pyplot.figure(figsize=(18, 6))
    pyplot.subplot(131)
    q1 = pyplot.quiver(0, 0, angles='xy', scale_units='xy', scale=1, width=0.003, color='red')
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(132)
    q2 = pyplot.quiver(0, 0, angles='xy', scale_units='xy', scale=1, width=0.003, color='blue')
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(133)
    q3 = pyplot.quiver(0, 0, angles='xy', scale_units='xy', scale=1, width=0.003, color='magenta')
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])

    anim1 = animation.FuncAnimation(fig, update_quiver, fargs=(q1, x[0], x[1]), interval=50, blit=False)
    anim2 = animation.FuncAnimation(fig, update_quiver, fargs=(q2, y[0], y[1]), interval=50, blit=False)
    anim3 = animation.FuncAnimation(fig, update_quiver, fargs=(q3, z[0], z[1]), interval=50, blit=False)
    pyplot.show()
    return


def demo3():
    """
    Transformation of a singular matrix
    :return:
    """
    matrix = numpy.array([[4, 0], [4, 0]])
    matrix_inv = numpy.linalg.pinv(matrix)
    print(matrix_inv)

    x = unit_ball(p=numpy.inf, num_points=100)
    y = numpy.matmul(matrix, x)
    z = numpy.matmul(matrix_inv, y)
    max_lim = max(numpy.abs(x).max(), numpy.abs(y).max())

    fig = pyplot.figure(figsize=(18, 6))
    pyplot.subplot(131)
    q1 = pyplot.quiver(0, 0, angles='xy', scale_units='xy', scale=1, width=0.003, color='red')
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(132)
    q2 = pyplot.quiver(0, 0, angles='xy', scale_units='xy', scale=1, width=0.003, color='blue')
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])
    pyplot.subplot(133)
    q3 = pyplot.quiver(0, 0, angles='xy', scale_units='xy', scale=1, width=0.003, color='magenta')
    pyplot.xlim([-max_lim, max_lim])
    pyplot.ylim([-max_lim, max_lim])

    anim1 = animation.FuncAnimation(fig, update_quiver, fargs=(q1, x[0], x[1]), interval=50, blit=False)
    anim2 = animation.FuncAnimation(fig, update_quiver, fargs=(q2, y[0], y[1]), interval=50, blit=False)
    anim3 = animation.FuncAnimation(fig, update_quiver, fargs=(q3, z[0], z[1]), interval=50, blit=False)
    pyplot.show()
    return


def main():
    # demo1()
    # demo2()
    demo3()
    return


if __name__ == '__main__':
    main()
