import sys
import matplotlib
import pytest


if __name__ == "__main__":
    # This has to be called before any import of matplotlib.pyplot
    # to be able to run plot on Travis.
    matplotlib.use('agg')

    # default arguments
    args = ['--pyargs', 'pactools', '-v', '--duration=10']  # '--cov=pactools'
    # append the arguments of the command line
    for arg in sys.argv:
        args.append(arg)

    pytest.main(args)
