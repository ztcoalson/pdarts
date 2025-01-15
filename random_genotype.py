from collections import namedtuple
import random

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3', #
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5', #
    'dil_conv_3x3',
    'dil_conv_5x5'  #
]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def sample_random_genotype(num):
    for _ in range(num):
        normal_cell, reduce_cell = [], []
        for i in range(8):
            normal_cell.append((PRIMITIVES[random.randint(1, 7)], random.randint(0, 1+(i // 2))))
            reduce_cell.append((PRIMITIVES[random.randint(1, 7)], random.randint(0, 1+(i // 2))))
        print(Genotype(normal=normal_cell, normal_concat=range(2,6), reduce=reduce_cell, reduce_concat=range(2,6)))

if __name__ == '__main__':
    sample_random_genotype(4)