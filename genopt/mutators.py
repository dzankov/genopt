import random
from copy import deepcopy
from .utils import flip_coin


def uniform_mutation(individual, space, prob=0):
    return individual