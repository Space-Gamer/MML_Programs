# Unit balls in 2D
import numpy as np
import matplotlib.pyplot as plt
import P01_Norms as Norms


def plot_unit_ball(p: int = 2):
    # theta = numpy.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120, 150, 180, 235, 270, 315]) * numpy.pi / 180
    # x = numpy.zeros(theta.shape) + 1
    # y = numpy.tan(theta)
    theta = np.linspace(0, 360, num=1000) * np.pi / 180
    x = np.cos(theta)
    y = np.sin(theta)
    vector = np.stack([x, y], axis=1)
    # vector_norm = Norms.vector_norm(vector, p) # Semantic error
    vector_norm = np.linalg.norm(vector, p, axis=1, keepdims=True)
    unit_vector = vector / vector_norm

    plt.figure(figsize=(5, 5))
    plt.plot(unit_vector[:, 0], unit_vector[:, 1])
    plt.show()
    return


def main():
    # for p in range(1, 1000, 100):
    #     plot_unit_ball(p)
    plot_unit_ball(p=2)
    return


if __name__ == '__main__':
    main()
