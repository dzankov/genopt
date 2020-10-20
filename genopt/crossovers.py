import random
from copy import deepcopy


def one_point_crossover(mother, father):
    sister = deepcopy(mother)
    brother = deepcopy(father)

    cut = random.randint(1, len(mother) - 1)

    sister[cut:] = father[cut:]
    brother[cut:] = mother[cut:]

    return sister, brother


def two_point_crossover(mother, father):
    sister = deepcopy(mother)
    brother = deepcopy(father)

    cut1 = random.randint(1, len(mother) - 1)
    cut2 = random.randint(1, len(mother) - 1)

    if cut1 == cut2:
        cut2 = random.randint(1, len(mother) - 1)

    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    sister[0:cut1], sister[cut2:] = father[0:cut1], father[cut2:]
    brother[0:cut1], brother[cut2:] = mother[0:cut1], mother[cut2:]

    return sister, brother


def uniform_crossover(mother, father, indpb=0.1):
    sister = deepcopy(mother)
    brother = deepcopy(father)

    for i in range(len(sister)):
        if random.random() < indpb:
            slot = sister[i]
            sister[i] = brother[i]
            brother[i] = slot

    return sister, brother


def gen_calc(x, y, alpha):
    gen = alpha * x + (1 - alpha) * y
    return gen


def whole_arithmetic_crossover(mother, father):
    sister = deepcopy(mother)
    brother = deepcopy(father)
    alpha = random.random()
    for i in range(len(mother)):
        sister[i] = gen_calc(mother[i], father[i], alpha)
        brother[i] = gen_calc(father[i], mother[i], alpha)
    return sister, brother

