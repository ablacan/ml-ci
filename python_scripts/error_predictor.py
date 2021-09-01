from __future__ import division # always-float division
import numpy as np
import pandas as pd
import glob
import os
import requests
from datetime import date

#Import from utils
from utils import *

from scipy.optimize import minimize
import scipy.stats as stats

#Import data generator
from generator_script import Generator

#Import epidemiological model, either SIR or SEIR
from epi_model import EPI_Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.inspection import permutation_importance


class ErrorPredictor():

    def __init__(self, model='SIR', PARAMS_FILE='None'):
        self.model = model
        self.PARAMS_FILE = PARAMS_FILE
        self.file_model = model
        self.file_method = 'results_error_predictor'

        #Create directories for future results
        if not os.path.exists('./{}'.format(self.file_method)):
            os.makedirs('./{}'.format(self.file_method))

        if not os.path.exists('./{0}/{1}'.format(self.file_method, self.file_model)):
            os.makedirs('./{0}/{1}'.format(self.file_method, self.file_model))


    #Generate training data
    @staticmethod
    def get_data(samples):
        """
        Returns training set of stringency and mobility data to use in simulations of covid19 cases.
        """
        Generator = Generator()
        stringency = Generator.generate(samples, type='stringency')
        #mobility = Generator_mobility.generate(samples, type = 'mobility')

        beta = Generator.generate(samples, type='beta')

        return stringency, mobility, beta


    #Generate simulations of cases according to beta
    def simulate(self, samples):
        """
        Returns simulated cases from EPI_Model
            samples : number of simulations, int
        """

        Model = EPI_Model(model=self.model, beta_step = 'linear')

        #Get data for simulation
        stringency, mobility, beta = self.get_data(samples)
        #Generate simulations with respective population
        simulations, population = Model.simulations(beta, samples)

        return simulations, population, stringency, mobility, beta


    #Train epi_model
    def fit(self, data, stringency, population, beta, train_window=100, samples=100, SAVE_TO_PATH=None):
        """
        Fit model on simulations
            train_window : number of days to train the epi model, int
        """
        self.train_window = train_window

        Model = EPI_Model(model=self.model, beta_step = 'stringency', PARAMS_FILE='./params_new/params_countries/params_SIR_countries_linear_stringency.csv')
        Model.train_window = train_window
        Model.test_window = 15
        #Get simulations
        #covid_cases, population, stringency_data, mobility_data, beta = self.simulate(samples)

        self.results_dict = []

        for i in range(len(data)):

            #Create dataframe
            df = pd.DataFrame(data={'date' : pd.date_range("2020", freq="D", periods=len(data[i])), 'I': data[i]})
            country_attrs = {'population' : population[i]}

            Model.stringency_country = stringency[i]

            #Prepare training with fixed gamma, sigma (disease_vary=False)
            train_data, test_data, params, initial_conditions = Model.prepare_training(df, country_attrs, disease_vary=False)

            #Training and forecast
            tspan_train = np.arange(1, 1+self.train_window, 1)
            ci, train_data, test_data, result, train_final, MSE, MAE, sol, test_MSE, test_MAE = Model.fit_test(df, country_attrs, disease_vary=False)

            #Save dictionary of results for each simulation
            self.results_dict.append({'population' : country_attrs['population'], 'ci' :ci, 'result': result, 'train_data': train_data,  'stringency_data' : stringency[i], 'beta': beta[i],  'train_fit': train_final,
            'test_data': test_data, 'test_predicted': sol})

            #Saving results
            if SAVE_TO_PATH != None:
                save_to_path = SAVE_TO_PATH
                try:
                    results_dict = np.load(save_to_path, allow_pickle=True)
                    results_dict = results_dict.tolist()
                    results_dict.append({'population' : country_attrs['population'], 'ci' :ci, 'result': result, 'train_data': train_data,  'stringency_data' : stringency[i], 'beta': beta[i],  'train_fit': train_final,
                    'test_data': test_data, 'test_predicted': sol})
                    np.save(save_to_path, results_dict)

                except FileNotFoundError:
                    np.save(save_to_path, self.results_dict)

            elif SAVE_TO_PATH==None:
                save_to_path = './{0}/{1}/results_simulations_{2}samples_{3}days.npy'.format(self.file_method, self.file_model, samples, train_window)
                try:
                    results_dict = np.load(save_to_path, allow_pickle=True)
                    results_dict = results_dict.tolist()
                    results_dict.append({'population' : country_attrs['population'], 'ci' :ci, 'result': result, 'train_data': train_data,  'stringency_data' : stringency[i], 'beta': beta[i],  'train_fit': train_final,
                    'test_data': test_data, 'test_predicted': sol})
                    np.save(save_to_path, results_dict)

                except FileNotFoundError:
                    np.save(save_to_path, self.results_dict)


    #Get training inputs for error predictor model
    def prepare_inputs(self, RES, prediction_horizon=1, error_bar='cumulative'):
        """
        Returns tuple of target and training inputs for error predictor model for a given prediction horizon.
            RES : dictionnary of results previoulsy fitted, dict
            prediction_horizon : days ahead to predict error on, int (>0)
        """
        assert prediction_horizon !=0, 'Error: prediction_horizon must be between 15 and 1.'

        #Add nb_lookback_days ??

        if error_bar=='cumulative':
            #Past inputs
            past_cases = RES['train_fit']
            past_derivative_true = np.concatenate((np.zeros(1).reshape(1,), np.diff(RES['train_data'], 1)), axis=0)
            past_derivative = np.concatenate((np.zeros(1).reshape(1,), np.diff(past_cases, 1)), axis=0)
            past_error = np.abs(RES['train_data'] - RES['train_fit'])
            past_error2 = past_error**2
            past_error_relative = np.abs(past_derivative_true - past_derivative)/(past_derivative_true+0.001)
            past_error_relative[0] = 0.0 #Remove NaN
            past_error_relative2 = past_error_relative**2
            past_stringency = RES['stringency_data'].ravel()[:self.training_window]

            if self._predict:
                bias_param = RES['result'].params['b'].value
                coeff_param = RES['result'].params['w1'].value
                #Compute beta rate
                past_beta = coeff_param*past_stringency + bias_param

            elif not self._predict:
                past_beta =  RES['beta'].ravel()[:self.training_window]

            #Second derivative
            past2nd_derivative = np.concatenate((RES['train_fit'][1:] , np.zeros(1).reshape(1,)), axis=0) -2*RES['train_fit'] + np.concatenate((np.zeros(1).reshape(1,), RES['train_fit'][:-1] ), axis=0)
            past2nd_derivative_true = np.concatenate((RES['train_data'][1:] , np.zeros(1).reshape(1,)), axis=0) -2*RES['train_data'] + np.concatenate((np.zeros(1).reshape(1,), RES['train_data'][:-1] ), axis=0)

            self.deriv2_true.append(past2nd_derivative_true)
            self.deriv2.append(past2nd_derivative)

            #Target at different horizons
            targets = np.abs(RES['test_data'] - RES['test_predicted'])

            #Get target at horizon i
            target = np.abs(targets[prediction_horizon-1])

            #if prediction_horizon ==1:

            inputs = np.stack((past_cases, past_derivative, past_derivative_true, past_error, past_error2, past_error_relative, past_error_relative2, past2nd_derivative, past2nd_derivative_true,
                                past_stringency, past_beta), axis=0)
            #print(past_derivative)
            #print(past_derivative_true)
            #print(past_error_relative)
            #print(past2nd_derivative)
            #print(past2nd_derivative_true)
            #return (target, inputs0)

        elif error_bar=='derivative':
            #Past inputs
            past_cases = RES['train_fit']
            past_cases_true = RES['train_data']
            past_derivative_fit = np.concatenate((np.zeros(1).reshape(1,), np.diff(past_cases, 1)), axis=0)
            past_derivative_true = np.concatenate((np.zeros(1).reshape(1,), np.diff(past_cases_true, 1)), axis=0)

            past_error = np.abs(past_derivative_true - past_derivative_fit)
            past_error2 = past_error**2

            past_stringency = RES['stringency_data'].ravel()[:self.training_window]
            past_beta =  RES['beta'].ravel()[:self.training_window]

            #Target at different horizons
            test_cases_true = RES['test_data']
            test_cases_preds = RES['test_predicted']
            test_derivative_preds = np.concatenate((past_derivative_fit[-1].reshape(1,), np.diff(test_cases_preds, 1)), axis=0)
            test_derivative_true = np.concatenate((past_derivative_true[-1].reshape(1,), np.diff(test_cases_true, 1)), axis=0)

            targets = np.abs(test_derivative_true - test_derivative_preds)

            #Get target at horizon i
            target = np.abs(targets[prediction_horizon-1])

            #if prediction_horizon ==1:
            inputs = np.stack((past_cases, past_derivative_fit, past_error, past_error2, past_stringency, past_beta), axis=0)

        elif error_bar =='relative':
            #Past inputs
            past_cases = RES['train_fit'][1:]
            past_derivative_true =  np.diff(RES['train_data'], 1)
            past_derivative_fit =  np.diff(RES['train_fit'], 1)
            past_error = np.abs(RES['train_data'][1:] - RES['train_fit'][1:])/past_derivative_true
            past_error2 = past_error**2
            past_stringency = RES['stringency_data'].ravel()[1:self.training_window]
            past_beta =  RES['beta'].ravel()[1:self.training_window]

            #Target at different horizons
            test_derivative_true =  np.concatenate(( past_derivative_true[-1].reshape(1,) , np.diff(RES['test_data'], 1)), axis=0)
            targets = np.abs(RES['test_data'] - RES['test_predicted'])/test_derivative_true

            #Get target at horizon i
            target = np.abs(targets[prediction_horizon-1])

            #if prediction_horizon ==1:
            inputs = np.stack((past_cases, past_derivative_fit, past_error, past_error2, past_stringency, past_beta), axis=0)

            #elif prediction_horizon >1:
                #Stack inputs
                #past_cases = np.stack((past_cases,  RES['test_data'][:prediction_horizon-2]), axis=0)
                #past_derivative = np.stack((past_cases), 1)
                #past_error = np.stack((past_error, targets[:prediction_horizon-2]))
                #past_error2 = np.stack((past_error2, targets[:prediction_horizon-2]**2))

                #past_stringency = RES['stringency_data'][:250+prediction_horizon-1]
                #past_beta =  RES['beta'][:250+prediction_horizon-1]

                #inputs = np.stack((past_cases, past_derivative, past_error, past_error2, past_stringency, past_beta), axis=0)
        return (target, inputs)

    #Negative log likelihood function to minimize
    @staticmethod
    def neg_LL(y, yhat, std):
         #intercept, beta, std = params[0], params[1], params[2] #initial parameters
        # Then we compute PDF of observed values normally distributed around mean (yhat) with a standard deviation of std
         negLL = -np.sum(stats.norm.logpdf(y, loc=yhat, scale=std) )# return negative LL
         #Append loss to evaluate convergence
         #self._loss.append(negLL)
         return negLL

    @staticmethod
    def compute_likelihood(y, yhat, sigma, calibration=False):
        T = len(y)
        num = np.abs(y - yhat)**2
        negLL = 0.5*np.sum(num/sigma**2) + np.sum(np.log(sigma)) + T*np.log(np.sqrt(2*pi))

        return negLL



    #Train error predictor model on given time horizon
    def train(self, samples=1000, train_window = 100, prediction_horizon=1, error_bar='cumulative', FROM_PATH=None):
        """
        Returns random forest regressor predictions, target values, test predictions and target values, and R2 training score.
            prediction_horizon : horizon of days to predict error on, int (>0).
        """
        self.training_window = train_window

        #Fit epi_model on a number of simulations=samples
        #self.fit(samples, train_window = train_window)

        #Load results previously fitted
        if FROM_PATH==None:
            from_path = './{0}/{1}/results_simulations_{2}samples_{3}days.npy'.format(self.file_method, self.file_model, samples, train_window)
        elif FROM_PATH != None:
            from_path = FROM_PATH
        RES = np.load(from_path, allow_pickle=True)

        #Get target and inputs for error predictor training given a prediction_horizon and a dictionnary of results
        #Init
        self.deriv2_true = []
        self.deriv2 = []
        target, x = self.prepare_inputs(RES[0], prediction_horizon, error_bar=error_bar)
        X = x.flatten().reshape(1, x.shape[0]*x.shape[1])
        Y = np.asarray([target]).reshape(1,)

        #Loop over all dicts of results
        for res_dict in RES[1:len(RES)]:
            target, x = self.prepare_inputs(res_dict, prediction_horizon, error_bar=error_bar)
            #print('x', x.shape)
            #print('X', X.shape)
            X = np.concatenate((X, x.reshape(1, x.shape[0]*x.shape[1])), axis=0)
            Y = np.concatenate((Y, np.asarray(target).reshape(1,)))

        #Split train_test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

        #Train the error predictor model
        RF_Model = RandomForestRegressor(n_estimators = 500, max_depth=20, random_state=42)
        RF_Model.fit(X_train, y_train)

        #init_params = np.array([...])
        #model = minimize(neg_LL, init_params, args=(X_train, y_train) method = 'Nelder-Mead', options={'disp': True, 'maxiter': 500})

        # Save model
        #np.save(model.x,'./{0}/{1}/Error_Predictor_Model_{3}samples_{4}days.npy'.format(self.file_method, self.file_model, samples, train_window))
        joblib.dump(RF_Model,'./{0}/{1}/Error_Predictor_Model_{2}samples_{3}days_horizon{4}_{5}.npy'.format(self.file_method, self.file_model, samples, train_window, prediction_horizon, error_bar))

        #Features importance
        #self._feature_names = ['past_cases', 'past_derivative', 'past_derivative_true', 'past_error', 'past_error2', 'past_error_relative', 'past_error_relative2', 'past2nd_derivative',
        #'past2nd_derivative_true', 'past_stringency', 'past_beta']
        #res_importance = permutation_importance(RF_Model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        #forest_importances = pd.Series(res_importance.importances_mean, index=self._feature_names)

        #Predict on test samples
        train_predictions = RF_Model.predict(X_train)
        test_predictions = RF_Model.predict(X_test)
        r2_train  = RF_Model.score(X_train, y_train)
        r2_test  = RF_Model.score(X_test, y_test)

        return train_predictions, y_train, test_predictions, y_test, r2_train, r2_test#, forest_importances


    def predict(self, RES, model_path, prediction_horizon=1, error_bar='cumulative'):
        """
        Returns error predictions on input samples given a horizon of prediction.
            inputs : dict of results from epidemiological model, list of dicts
            model_path : path from which to load the error predictor model, str
            prediction_horizon : error prediction horizon, int
        """

        #Init
        self._predict=True
        self.deriv2_true = []
        self.deriv2 = []

        #Prepare inputs
        target, x = self.prepare_inputs(RES[0], prediction_horizon, error_bar=error_bar)
        X = x.flatten().reshape(1, x.shape[0]*x.shape[1])
        Y = np.asarray([target]).reshape(1,)

        #Loop over all dicts of results
        if len(RES)>1:
            for res_dict in RES[1:len(RES)]:
                target, x = self.prepare_inputs(res_dict, prediction_horizon, error_bar=error_bar)
                #print('x', x.shape)
                #print('X', X.shape)
                X = np.concatenate((X, x.reshape(1, x.shape[0]*x.shape[1])), axis=0)
                Y = np.concatenate((Y, np.asarray(target).reshape(1,)))

        # Load
        #params = np.load(model_path, allow_pickle=True)
        RF_Model = joblib.load(model_path)

        #Predict on samples
        predictions = RF_Model.predict(X)
        #predictions = np.matmul(X, params)

        return predictions, Y
