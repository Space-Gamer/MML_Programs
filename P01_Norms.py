import numpy


def vector_norm(x, p=2):
    x_norm = 0
    for x_i in x:
        x_norm += (abs(x_i) ** p)
    x_norm = x_norm ** (1 / p)
    return x_norm


def main():
    x = numpy.array([1, 2, -3, 4])
    p = 1
    x_norm = vector_norm(x, p)
    print(x)
    print(x_norm)
    print(numpy.linalg.norm(x, p))
    return


if __name__ == '__main__':
    main()
