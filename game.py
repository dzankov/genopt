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
from pickle import dump
from time import time


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
logger.info(f" Program started. Target: {str(target)}, Steps: {steps}, "
            f"Population size: {popsize}, Number of steps: {steps}")

play = SimpleSynthesis(target, steps=10 ** 6)

ga = SGA(task='maximize', pop_size=popsize, cross_prob=0.9, mut_prob=0.1, elitism=True, n_cpu=10)
ga.set_selector_type(tournament_selection)
ga.set_scaler_type(sigma_trunc_scaling)
ga.set_crossover_type(two_point_crossover)
ga.set_mutator_type(uniform_mutation)


def fitness(chromo):
    start = time()
    reward = 0
    play.reset()
    for step in chromo:
        state, reward, done, info = play.step(int(step))
        if state:
            if done:
                return reward
            if len(state) - len(target) >= 20:
                play.reset()
    print(str(time() - start))

    return reward


ga.set_fitness(fitness)

pop_dict = {}
# init action space and population
ga.initialize(play.action_space, chromo_len)
pop_dict['init'] = ga.population
with open(pop_dict_name, 'wb') as f:
    dump(pop_dict, f)

tanimoto = 0
for i in range(steps):
    logger.info(f'\nstep {i} started')
    ga.step()
    pop_dict[i] = ga.population
    with open(pop_dict_name, 'wb') as f:
        dump(pop_dict, f)
    chromo = ga.best_individual()
    stat = ga.population.calc_statistics()
    stat.stats['chromo'] = chromo
    tanimoto = chromo.score
    for k in stat.stats:
        logger.info(f'{k}:, {stat.stats[k]}')
    if tanimoto >= 10:
        logger.info('DONE')
        for r in play.render():
            logger.info(r)
        break


for r in play.render():
    logger.info(r)
