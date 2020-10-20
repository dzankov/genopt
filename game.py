from config import *
from genopt.crossovers import two_point_crossover
from genopt.mutators import uniform_mutation
from genopt.optimizer import SGA
from genopt.selectors import tournament_selection
from genopt.scalers import sigma_trunc_scaling
from genopt.utils import weighted_sample_without_replacement
from CGRtools.files import SDFRead
import logging
import os

os.environ['DATA'] = '/home/adelia/rules'
os.environ['DB_NAME'] = 'postgres'
os.environ['DB_USER'] = user
os.environ['DB_PASS'] = password
os.environ['DB_HOST'] = host
os.environ['DB_SCHEMA'] = db
os.environ['DB_PORT'] = port
os.environ['PATH'] += '/home/adelia/ga/bin'

from RNNSynthesis.environment import SimpleSynthesis


target = next(SDFRead(target))


play = SimpleSynthesis(target, steps=10 ** 6)

ga = SGA(task='maximize', pop_size=5, cross_prob=0.8, mut_prob=0.1, elitism=True)
ga.set_selector_type(tournament_selection)
ga.set_scaler_type(sigma_trunc_scaling)
ga.set_crossover_type(two_point_crossover)
ga.set_mutator_type(uniform_mutation)


def f(chromo):
    r = None
    play.reset()
    for step in chromo:
        s, r, d, info = play.step(int(step))
        if len(s) - len(target) >= 20:
            play.reset()
        # print('state', s, 'reward', r, 'info', info)
    return r


ga.set_fitness(f)


# init action space and population
ga.initialize(play.action_space)

tan = 0
for i in range(steps):
    ga.step()
    chromo = ga.best_individual()
    stat = ga.population.calc_statistics()
    stat.stats['chromo'] = chromo
    tan = chromo.score
    for k in stat.stats:
        print(k, stat.stats[k])
    if tan >= 10:
        print('DONE')
        break
