import pathlib
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', default=0, type=int, nargs='?',
                    help='Seed for the random number generator')
parser.add_argument('--coordinates-file', required=True, type=pathlib.Path, nargs='?',
                    help='File path for the coordinates file')
parser.add_argument('--orders-file', required=True, type=pathlib.Path, nargs='?',
                    help='File path for the orders file')
parser.add_argument('--num-pickers', default=2, type=int, nargs='?',
                    help='Number of pickers')
parser.add_argument('--out-file', type=pathlib.Path, nargs='?',
                    help='Output file path')

# batch orders parameters
parser.add_argument('--l-batch-orders', default=50, type=int, nargs='?',
                    help='L batch orders')
parser.add_argument('--k2-batch-orders', default=5, type=int, nargs='?',
                    help='k2 batch orders')
parser.add_argument('--epsilon2-batch-orders', default=10, type=float, nargs='?',
                    help='eplison2 batch orders')
parser.add_argument('--initial-temperature-batch-orders', default=50, type=int, nargs='?',
                    help='Initial temperature value for the simulated annealing (batch orders)')
parser.add_argument('--cooling-rate-batch-orders', default=0.9, type=float, nargs='?',
                    help='Cooling rate value for the simulated annealing (batch orders)')

# order items parameters
parser.add_argument('--l-order-items', default=50, type=int, nargs='?',
                    help='L order items')
parser.add_argument('--k2-order-items', default=5, type=int, nargs='?',
                    help='k2 order items')
parser.add_argument('--epsilon2-order-items', default=10, type=float, nargs='?',
                    help='epsilon2 order items')
parser.add_argument('--initial-temperature-order-items', default=50, type=int, nargs='?',
                    help='Initial temperature value for the simulated annealing (order items)')
parser.add_argument('--cooling-rate-order-items', default=0.9, type=float, nargs='?',
                    help='Cooling rate value for the simulated annealing (order items)')

# orders parameters 
parser.add_argument('--l-orders', default=20, type=int, nargs='?',
                    help='L orders')
parser.add_argument('--k2-orders', default=5, type=int, nargs='?',
                    help='k2 orders')
parser.add_argument('--epsilon2-orders', default=100, type=float, nargs='?',
                    help='epsilon2 orders')
parser.add_argument('--initial-temperature-orders', default=2, type=int, nargs='?',
                    help='Initial temperature value for the simulated annealing (orders)')
parser.add_argument('--cooling-rate-orders', default=0.9, type=float, nargs='?',
                    help='Cooling rate value for the simulated annealing (orders)')
