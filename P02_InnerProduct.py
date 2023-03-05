import numpy


def inner_product(x, y):
    inner_prod = 0
    for x_i, y_i in zip(x, y):
        inner_prod += (x_i * y_i)
    return inner_prod


def main():
    x = numpy.array([1, 2, 3])
    y = numpy.array([2, 0, 1])
    print(inner_product(x, y))
    print(numpy.dot(x, y))
    return


if __name__ == "__main__":
    main()
