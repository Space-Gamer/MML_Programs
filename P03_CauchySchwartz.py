import numpy
import P01_Norms as Norms
import P02_InnerProduct as InnerProduct


def main():

    num_simulations = 1000

    # x = numpy.array([1, 2, 3])
    # y = numpy.array([4, 0, 5])

    for _ in range(num_simulations):

        dimension = numpy.random.randint(1, 100)
        x = numpy.random.rand(dimension)
        y = numpy.random.rand(dimension)

        x_norm = Norms.vector_norm(x, p=2)
        y_norm = Norms.vector_norm(y, p=2)
        xy_ip = InnerProduct.inner_product(x, y)

        if numpy.abs(xy_ip) > x_norm * y_norm:
            print(f'Cauchy Schwartz not satisfied for x = {x} and y = {y}')
            return

    print('Cauchy Schwartz satisfied for all simulations')
    # print(x)
    # print(y)
    # print('|<x,y>|:', xy_ip)
    # print('||x||:', x_norm)
    # print('||y||:', y_norm)
    # if numpy.abs(xy_ip) <= x_norm * y_norm:
    #     print('Cauchy Schwartz satisfied')
    # else:
    #     print('Error!')
    # return


if __name__ == '__main__':
    main()
