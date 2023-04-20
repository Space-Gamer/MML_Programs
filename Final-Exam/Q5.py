# Write a python program to perform the following geometric operations through a linear transformation y = Ax in R2.
# (a) Reflection on y-axis
# (b) Scale the length of vector by 3 times and rotate it by 60o
# .
# Visualize the input and transformed vector in each case for the following vectors.

# (a) [1, 0]
# (b) [0, 1]
# (c) [1, 1]
# (d) [-1, 2]

import numpy as np
import matplotlib.pyplot as plt


def visualize_linear_transform(matrix, vector):
    # Create a plot to visualize the transformation
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    # Apply the transformation to the vector
    transformed_vector = np.dot(matrix, vector)

    # Plot the vector and transformed vector
    ax.quiver([0], [0], [vector[0]], [vector[1]], color=['b'], angles='xy', scale_units='xy', scale=1, label='Original')
    ax.quiver([0], [0], [transformed_vector[0]], [transformed_vector[1]], color=['r'], angles='xy', scale_units='xy', scale=1, label='Transformed')
    ax.legend()
    plt.show()

    return transformed_vector


def rotation_matrix(vector, theta):
    theta = theta * np.pi / 180
    # Create a rotation matrix
    matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return matrix


def main():

    vector1 = np.array([1, 0])
    vector2 = np.array([0, 1])
    vector3 = np.array([1, 1])
    vector4 = np.array([-1, 2])

    # (a) Reflection on y-axis
    matrix1 = np.array([[-1, 0], [0, 1]])
    print('Reflection on y-axis for vector1')
    print('Reflected vector is: ', visualize_linear_transform(matrix1, vector1))

    # (b) Scale the length of vector by 3 times and rotate it by 60o
    matrix2 = np.dot(rotation_matrix(vector1, 60), np.diag([3, 3]))
    print('Scale the length of vector by 3 times and rotate it by 60o for vector1')
    print('Transformed vector is: ', visualize_linear_transform(matrix2, vector1))

    print('Reflection on y-axis for vector2')
    print('Reflected vector is: ', visualize_linear_transform(matrix1, vector2))

    print('Scale the length of vector by 3 times and rotate it by 60o for vector2')
    print('Transformed vector is: ', visualize_linear_transform(matrix2, vector2))

    print('Reflection on y-axis for vector3')
    print('Reflected vector is: ', visualize_linear_transform(matrix1, vector3))

    print('Scale the length of vector by 3 times and rotate it by 60o for vector3')
    print('Transformed vector is: ', visualize_linear_transform(matrix2, vector3))

    print('Reflection on y-axis for vector4')
    print('Reflected vector is: ', visualize_linear_transform(matrix1, vector4))

    print('Scale the length of vector by 3 times and rotate it by 60o for vector4')
    print('Transformed vector is: ', visualize_linear_transform(matrix2, vector4))


if __name__ == '__main__':
    main()
