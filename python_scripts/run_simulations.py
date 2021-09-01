#Script for simulating covid19 counterfactuals

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import argparse

from generator_script import Generator
from epi_model import EPI_Model
from error_predictor import ErrorPredictor
import matplotlib.pyplot as plt
import random
#random.seed(1)

parser = argparse.ArgumentParser()

parser.add_argument("-nb_simulations", "--number_of_simulations",
                        dest="nb_simulations",
                        type=int,
                        required=True,
                        help="Specify number of simulations to generate >0 (required)")
args = parser.parse_args()


"""Generate stringency and beta simulations"""
SIMULATIONS = args.nb_simulations

#Init generator
GENERATOR = Generator()
print('Generate stringency and beta...')
simulations, beta_simulations = GENERATOR.generate(SIMULATIONS)

"""Generate counterfactuals with SIR model"""
EPI_Model = EPI_Model(model='SIR')
print('Generate counterfactuals and save simulations...')
cases_simulations, populations = EPI_Model.simulations(beta_simulations, SIMULATIONS)
#SAVE
FILE_METHOD = 'results_error_predictor'
FILE_MODEL = 'SIR'
np.save('./{0}/{1}/simulations_{2}samples.npy'.format(FILE_METHOD, FILE_MODEL, SIMULATIONS), {'simulations':simulations, 'betas' : beta_simulations, 'cases' : cases_simulations, 'populations': populations})


"""Fit SIR model on simulations"""
Predictor = ErrorPredictor(model='SIR')
print('Fitting SIR on simulations...')
Predictor.fit(cases_simulations, simulations, populations, beta_simulations, train_window=100, samples=SIMULATIONS)
