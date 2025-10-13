import argparse
from jax_mppi import nmppi_global_random
from jax_mppi import mppi_random

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', nargs='?', default="ipopt", help='filename')
    args = parser.parse_args()
    mppi_random.main(args)
