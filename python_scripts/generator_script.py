from __future__ import division # always-float division
import numpy as np
import os
import requests
from time import process_time
import random

random.seed(1)

class Generator:
    """
    Class for generating stringency data : to use in the training of our model to predict systematic errors.
    """
    def __init__(self):
        pass


    @staticmethod
    def logistic(x, L, k, x0):
        """
        Logistic function on 'x' with 'L' maximum value, 'k' steepness and 'x0' midpoint
        """
        return L/(1+np.exp(-k*(x-x0)))


    def logistic_step(self, t, params):
        """
        Not-0 baseline logistic function with 'params'=[C, L, a, b] where
            C initial value
            C+L final value
            a step steepness (a>0)
            b step midpoint
        """
        C, L, a, b = params
        return C + self.logistic(t, L, a, b)

    @staticmethod
    def normalize_data(data):
        """Normalize data between 0.0 and 1.0"""
        return (data - data.min())/(data.max() - data.min())

    @staticmethod
    def generate_sigmoid(NB_SIMUL):
        """
        Returns one step sigmoid parameters C, L, a, b with random sampling.
            NB_SIMUL : nb of simulations, int
        """
        C_rand = np.zeros(NB_SIMUL) #Starting the curve at 0 because almost no measures implemented at the beginning of 2020.
        L_rand = [np.random.uniform(0.0, 1.0) for C in C_rand]
        # L_rand = [np.random.uniform(0.0, 1-C) for C in C_rand]
        a_rand = np.random.uniform(size=NB_SIMUL)
        b_rand = np.random.randint(1, 150, NB_SIMUL) #(maximum day of lag ~150 for the first wave)

        return np.stack((C_rand, L_rand, a_rand, b_rand), axis=1)

    @staticmethod
    def generate_ascent_descent(NB_SIMUL, sigmoid_params, MAX_DAYS, IDX_LAG=3, i=1):
        """
        Returns first sigmoid step parameters + second step sigmoid parameters (either ascent or descent)
            NB_SIMUL : nb of simulations, int
            sigmoid_params : previous sigmoid step parameters, np.array
            MAX_DAYS : maximum days for the lag sampling (parameter b), int
            IDX_LAG : index of previous sigmoid step lag parameter (parameter b)
            i : index of sigmoid step (4 steps in total), helps define if ascent or descent, int
        """
        #Second lag > to first lag at least by 8 days (1 week to implement measures)
        b_rand = [np.random.randint(b+7, max(b+8, MAX_DAY)) for b, MAX_DAY in zip(sigmoid_params[:,IDX_LAG], MAX_DAYS)]
        a_rand = (-1)**i * np.random.uniform(size=NB_SIMUL)
        L_rand = [max(np.random.uniform(0, C+L,1).item(), 0) for C,L in zip(sigmoid_params[:,IDX_LAG-3], sigmoid_params[:,IDX_LAG-2])]

        if (-1)**i>0: #Ascent
            C_rand = sigmoid_params[:,IDX_LAG-3]
        elif (-1)**i<0: #Descent
            C_rand = [C+L - L2 for C,L,L2 in zip(sigmoid_params[:,IDX_LAG-3], sigmoid_params[:,IDX_LAG-2], L_rand)]

        second_sig_params = np.stack((C_rand, L_rand, a_rand, b_rand), axis=1)

        return np.concatenate((sigmoid_params, second_sig_params), axis=1)


    def generate_full_sigmoid(self, NB_SIMUL):
        """
        Returns 4x4 parameters (4 parameters for 4 sigmoid steps ; looking like 2 uniform functions with different amplitudes) x NB_SIMUL
            Maximum of days for simulation -> from january to December 2020 ~ 350j.
            NB_SIMUL : nb of simulations, int
        """
        #Generate 1 sigmoid parameters
        sig_params = self.generate_sigmoid(NB_SIMUL)

        #Generate random ranges of days from where to sample lag parameter (b parameter)
        random_MAX_DAYS1 = [np.random.randint(previous_lag+7, previous_lag+100) for previous_lag in sig_params[:,3]]
        random_MAX_DAYS2 = [np.random.randint(random_MAX_DAY1+7, max(352, random_MAX_DAY1+100)) for random_MAX_DAY1 in random_MAX_DAYS1]
        random_MAX_DAYS3 = [np.random.randint(random_MAX_DAY2+7, max(random_MAX_DAY2+8, 360)) for random_MAX_DAY2 in random_MAX_DAYS2]

        #Loop over sigmoid steps (3 steps + 1 previously computed)
        for i, MAX_DAYS, IDX_LAG in zip(range(1, 4), [random_MAX_DAYS1, random_MAX_DAYS2, random_MAX_DAYS3], [3, 7, 11]):
            sig_params = self.generate_ascent_descent(NB_SIMUL, sig_params, MAX_DAYS, IDX_LAG, i)

        return sig_params



    def generate(self, NB_SIMUL=100, add_noise=False):
        """Returns simulated stringency time series which is a noisy version of beta⁻1 (a*beta + b)"""

        t1 = process_time()

        simulations = [] #Init
        beta_simulations = []

        sigmoids_params = self.generate_full_sigmoid(NB_SIMUL) #Generate parameters for 4 sigmoid steps per simulation

        #Loop over number of simulations to compute stringency data
        for i in range(len(sigmoids_params)):
            log1 = self.logistic_step(np.arange(1,sigmoids_params[i,3],1), sigmoids_params[i,:4])
            log2 = self.logistic_step(np.arange(sigmoids_params[i,3],sigmoids_params[i,7],1), sigmoids_params[i,4:8])
            log3 = self.logistic_step(np.arange(sigmoids_params[i,7], sigmoids_params[i,11],1), sigmoids_params[i,8:12])
            log4 = self.logistic_step(np.arange(sigmoids_params[i,11],360+1,1), sigmoids_params[i,12:16])

            simulated_stringency = np.concatenate((log1, log2, log3, log4), axis=0) #Concatenate 4 sigmoid steps data

            #Add gaussian noise to data
            if add_noise:
                simulated_stringency = simulated_stringency + np.random.normal(0.0, 0.005, len(simulated_stringency))

            #Normalize data
            simulated_stringency = self.normalize_data(simulated_stringency)

            #Derive beta simulations
            beta_s = -simulated_stringency.reshape(len(simulated_stringency))
            beta_s_norm =  self.normalize_data(beta_s)

            #Append
            simulations.append(simulated_stringency)
            beta_simulations.append(np.random.uniform(0.001, 0.5,1)*beta_s_norm+np.random.uniform(0.001, 0.2, 1))

        simulations = np.stack(simulations)
        beta_simulations = np.stack(beta_simulations)


        t_end = process_time()
        print(f'Simulation took {round(t_end - t1, 5)} seconds to run.')

        return simulations, beta_simulations
