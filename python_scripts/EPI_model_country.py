from __future__ import division # always-float division
import numpy as np
import pandas as pd
import glob
import pprint
import os
import requests
from datetime import date

# Easy interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import corner

# Interactive plots in notebook
from IPython.display import HTML, Image, display
from ipywidgets.widgets import interact, IntSlider, FloatSlider, Layout, ToggleButton, ToggleButtons, fixed, Checkbox

# Maths
from scipy.integrate import odeint
import scipy.signal.windows as window
from sklearn.preprocessing import normalize
import scipy.stats as stats

# Long computations
from tqdm import tqdm_notebook as tqdm
import pickle

# Fitter
from lmfit import Model, Parameters, Parameter, report_fit, minimize
import lmfit
from lmfit.printfuncs import *

#Import from utils
from utils import *



class EPI_Model():
    """
    Class for fitting compartmental epidemiologic model (either SIR, SEIR, SEIRD)
    """
    def __init__(self, model='SIR', beta_step = 'linear', PARAMS_FILE=None, OWID_FILE=None, cutoff_date='2020-12-31'):

        #assert OWID_FILE != None, "Data file not specified."
        #assert PARAMS_FILE != None, "Parameters file not specified."
        assert beta_step == 'constant' or 'linear' or 'sigmoid', "Beta parameter model must be either constant, linear or sigmoid."
        assert model == 'SIR' or 'SEIR' or 'SEIRD', "Compartmental model must be SIR, SEIR or SEIRD."

        self.model = model
        self.beta_step = beta_step
        self.PARAMS_FILE = PARAMS_FILE
        self.owid_file = OWID_FILE
        self.cutoff_date = cutoff_date
        self.countries = ['Australia', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'India',
        'Israel', 'Italy', 'Japan', 'Luxembourg', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Russia', 'South Africa', 'Korea, Republic of', 'Spain', 'Sweden',
        'Taiwan', 'United Kingdom', 'United States']

        self.iso_code2 = ['AU', 'AT', 'BE', 'BG', 'CA', 'CH', 'DK', 'FI', 'FR', 'DE', 'GR', 'IN', 'IL', 'IT', 'JP', 'LU', 'NL', 'NO', 'PL', 'PT', 'RU', 'ZA',
                     'KR', 'ES', 'SE', 'TW', 'GB', 'US']

        self.test_on_optim = False
        self.file_method = 'results_{}'.format(self.beta_step)
        self.file_model = model

        #Create directories for future results
        if not os.path.exists('./{}'.format(self.file_method)):
            os.makedirs('./{}'.format(self.file_method))

        if not os.path.exists('./{0}/{1}'.format(self.file_method, self.file_model)):
            os.makedirs('./{0}/{1}'.format(self.file_method, self.file_model))

    """~~~~UTILS FUNCTIONS~~~~"""

    """Logistic functions for sigmoid fits"""
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
    def linear_step(s, mob, params):
        """ Simple linear step to approximate beta parameter with stringency and mobility as inputs.
        """
        weight1, weight2, bias = params
        return weight1*s + weight2*mob + bias

    @staticmethod
    def surface_beta(data_s, data_mob, weights):
        """To plot surface plot in 3d and assess sensibility to parameters"""
        weight1, weight2, bias = weights
        X, Y = np.meshgrid(np.linspace(0,1,len(data_s)), np.linspace(-1,1,len(data_mob)))
        Z = weight1*X + weight2*Y + bias
        return Z

    def ODE_(self, z, t, params, beta_series):
        """
        Derivatives for ODE solver
        """
        # Separate compartments
        if self.model=='SIR':
            S, I, R = z
            N = S +I + R

            gamma = params[0]
            beta = get_continuous(t-1.0, beta_series)

            #Compartment derivatives
            dSdt = -beta*S*I/N
            dIdt = beta*S*I/N - gamma*I
            dRdt = gamma*I

            return [dSdt, dIdt, dRdt]

        elif self.model=='SEIR':
            S, E, I, R = z
            N = S + E + I + R

            sigma, gamma = rate_params
            beta = get_continuous(t-1.0, beta_series)

            # Compartment derivatives
            dSdt = -beta*S*I/N
            dEdt = beta*S*I/N - sigma*E
            dIdt = sigma*E - gamma*I
            dRdt = gamma*I
            return [dSdt, dEdt, dIdt, dRdt]


    def simulate(self, beta):
        """
        Returns simulations of cases using beta parameter time series only. Begins at day 1.
        """

        tspan = np.arange(1, len(beta)+1, 1)
        #Random population in millions of inhabitants:
        population = 1000000*np.random.uniform(0.01, 10, 1).item()
        initE, initI, initR, initN = 1.0, 1.0, 0.0, population
        self.last_I_cumul = initI
        # initial Susceptible
        if self.model=='SIR':
            initS = initN - (initI + initR)
            initial_conditions = [initS, initI, initR]
            #params = [np.random.uniform(0.03, 0.2)]
            params = [0.07]
        elif self.model=='SEIR':
            initS = initN - (initE + initI + initR)
            initial_conditions = [initS, initE, initI, initR]
            params = [0.1, 0.07]
            #params = [np.random.uniform(0.05, 0.25), np.random.uniform(0.03, 0.2)]

        result = odeint(self.ODE_, initial_conditions, tspan, args=(params))

        #Compute cumulated I cases
        simulation = self.get_cumulated_cases(result, self.last_I_cumul, test_phase=False)

        return simulation, population

    def simulations(self, beta_series, samples):
        """
        Returns simulations of cases for each beta time series.
            beta_series : time series of beta parameter ppreviously generated, np.array
            samples : nb of simulations to perform, int
        """
        assert samples == len(beta_series), "Number of simulations do not correspond to length of beta series."

        simulations = np.array([[self.simulate(beta) for beta in beta_series]])

        return simulations[:,0], simulations[:,1]



    """SIR, SEIR models derivatives to solve."""
    def SIRD_derivs(self, z, t, rate_params, evo_params=None):
        """
        Derivatives for ODE solver
        """
        # Separate compartments
        if self.model=='SIR':
            S, I, R = z
            N = S +I + R

            #Get parameters
            if self.beta_step=='constant':
                beta, gamma = rate_params

            elif self.beta_step=='linear':
                gamma = rate_params
                beta_params = evo_params
                #Get continuous values of inputs for integration.
                strin_continuous =  get_continuous(t-1.0, self.stringency_country) #stringency data starts at index 0 so we remove 1
                mob_continuous =  get_continuous(t-1.0, self.mob_country)
                beta = self.linear_step(strin_continuous, mob_continuous, beta_params)

            elif self.beta_step =='sigmoid':
                gamma = rate_params
                # Define evolution rates
                beta_params = evo_params
                beta= self.logistic_step(t, beta_params)

            #Compartment derivatives
            dSdt = -beta*S*I/N
            dIdt = beta*S*I/N - gamma*I
            dRdt = gamma*I

            return [dSdt, dIdt, dRdt]

        elif self.model=='SEIR':
            S, E, I, R = z
            N = S + E + I + R
            #get parameters
            if self.beta_constant_step:
                beta, sigma, gamma = rate_params

            elif self.beta_linear_step:
                sigma, gamma = rate_params
                beta_params = evo_params
                strin_continuous =  get_continuous(t-1.0, self.stringency_country) #stringency data starts at index 0 so we remove 1
                mob_continuous =  get_continuous(t-1.0, self.mob_country)
                beta = self.linear_step(strin_continuous, mob_continuous, beta_params)

            elif self.beta_sigmoid_step:
                sigma, gamma = rate_params
                # Define evolution rates
                beta_params = evo_params
                beta= self.logistic_step(t, beta_params)

            # Compartment derivatives
            dSdt = -beta*S*I/N
            dEdt = beta*S*I/N - sigma*E
            dIdt = sigma*E - gamma*I
            dRdt = gamma*I
            return [dSdt, dEdt, dIdt, dRdt]


    def ode_solver(self, t, initial_conditions, params):
        """
        ODE solver.
        """
        if self.model=='SIR':
            initI, initR, initN = initial_conditions
            # initial Susceptible
            initS = initN - (initI + initR)
            # Make param lists
            if self.beta_step=='sigmoid':
                gamma = params['gamma'].value
                # beta and mu params
                C, L, a, b = params[f'C'].value, params[f'L'].value, params[f'a'].value, params[f'b'].value
                # Static params and param lists
                rate_params = gamma
                beta_params = [C, L, a, b]
                evo_params = beta_params
                res = odeint(self.SIRD_derivs, [initS,initI, initR], t, args=(rate_params, evo_params))
                return res

            elif self.beta_step=='linear':
                gamma = params['gamma'].value
                w1, w2, b = params[f'w1'].value, params[f'w2'].value, params[f'b'].value
                # Static params and param lists
                rate_params = gamma
                beta_params = [w1, w2, b]
                evo_params = beta_params
                res = odeint(self.SIRD_derivs, [initS,initI, initR], t, args=(rate_params, evo_params))
                return res

            elif self.beta_step=='constant':
                beta, gamma = params['beta'].value, params['gamma'].value
                # Static params and param lists
                rate_params = [beta, gamma]
                res = odeint(self.SIRD_derivs, [initS, initI, initR], t, args=(rate_params, [])) #args has to be in a tuple
                return res


        elif self.model == 'SEIR':
            initE, initI, initR, initN = initial_conditions
            # initial Susceptible
            initS = initN - (initE + initI + initR)
            if self.beta_step=='sigmoid':
                # Make param lists
                sigma, gamma = params['sigma'].value, params['gamma'].value
                # beta params
                C, L, a, b = params[f'C'].value, params[f'L'].value, params[f'a'].value, params[f'b'].value

                # Static params and param lists
                rate_params = [sigma, gamma]
                beta_params = [C, L, a, b]
                evo_params = beta_params
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR], t, args=(rate_params, evo_params))
                return res

            elif self.beta_step=='linear':
                # Make param lists
                sigma, gamma = params['sigma'].value, params['gamma'].value
                # beta params
                w1, w2, b = params[f'w1'].value, params[f'w2'].value, params[f'b'].value

                # Static params and param lists
                rate_params = [sigma, gamma]
                beta_params = [w1, w2, b]
                evo_params = beta_params
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR], t, args=(rate_params, evo_params))
                return res

            elif self.beta_step=='constant':
                # Make param lists
                beta, sigma, gamma = params['beta'].value, params['sigma'].value, params['gamma'].value
                # Static params and param lists
                rate_params = [beta, sigma, gamma]
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR], t, args=(rate_params, []))
                return res


    """Solver functions"""
    def init_sectors(self, disease_vary=True, initN=0, initI=1, initR=0, initE=0):
        """
        Makes initial conditions : one infected and parameter initial values for optimization.
            'disease_params' are initial values for sigma, gamma mu, nu
            'disease_vary' freezes disease_params for optimization
        """
        params = Parameters()
        init_params = pd.read_csv(self.PARAMS_FILE, sep=";",index_col='name', header=0, skipinitialspace=True)

        #Tests on optimization with frozen parameters on SIR model
        if self.test_on_optim:
            #Init conditions
            initial_conditions = [initI, initR, initN]

            if self.beta_step=='sigmoid':
                params.add(f'C', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=False)
                params.add(f'L', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                params.add(f'a', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=False)
                params.add(f'b', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

            elif self.beta_step=='linear':
                params.add(f'w1', value=init_params.at['w1', 'init_value'], min=init_params.at['w1', 'min'], max=init_params.at['w1', 'max'], vary=True)
                params.add(f'w2', value=init_params.at['w2', 'init_value'], min=init_params.at['w2', 'min'], max=init_params.at['w2', 'max'], vary=True)
                params.add(f'b', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=False)

            #Add gamma
            params.add('gamma', value=init_params.at['gamma', 'init_value'], min=init_params.at['gamma', 'min'], max=init_params.at['gamma', 'max'], vary=False)
            return params, initial_conditions


        elif not self.test_on_optim:
            if self.model=='SIR':
                #Init conditions
                initial_conditions = [initI, initR, initN]

                if self.beta_step=='sigmoid':
                    params.add(f'C', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=True)
                    params.add(f'L', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                    params.add(f'a', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=True)
                    params.add(f'b', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                elif self.beta_step=='linear':
                    params.add(f'w1', value=init_params.at['w1', 'init_value'], min=init_params.at['w1', 'min'], max=init_params.at['w1', 'max'], vary=True)
                    params.add(f'w2', value=init_params.at['w2', 'init_value'], min=init_params.at['w2', 'min'], max=init_params.at['w2', 'max'], vary=True)
                    params.add(f'b', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                if self.beta_step=='constant':
                    params.add('beta', value=init_params.at['beta', 'init_value'], min=init_params.at['beta', 'min'], max=init_params.at['beta', 'max'], vary=True)

                #Add gamma
                params.add('gamma', value=init_params.at['gamma', 'init_value'], min=init_params.at['gamma', 'min'], max=init_params.at['gamma', 'max'], vary=disease_vary)


            elif self.model=='SEIR':
                initial_conditions = [initE, initI, initR, initN]

                if self.beta_step=='sigmoid':
                    params.add(f'C', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=True)
                    params.add(f'L', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                    params.add(f'a', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=True)
                    params.add(f'b', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                elif self.beta_step=='linear':
                    params.add(f'w1', value=init_params.at['w1', 'init_value'], min=init_params.at['w1', 'min'], max=init_params.at['w1', 'max'], vary=True)
                    params.add(f'w2', value=init_params.at['w2', 'init_value'], min=init_params.at['w2', 'min'], max=init_params.at['w2', 'max'], vary=True)
                    params.add(f'b', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                if self.beta_step=='constant':
                    params.add('beta', value=init_params.at['beta', 'init_value'], min=init_params.at['beta', 'min'], max=init_params.at['beta', 'max'], vary=True)

                #Add sigma and gamma
                params.add('sigma', value=init_params.at['sigma', 'init_value'], min=init_params.at['sigma', 'min'], max=init_params.at['sigma', 'max'], vary=disease_vary)
                params.add('gamma', value=init_params.at['gamma', 'init_value'], min=init_params.at['gamma', 'min'], max=init_params.at['gamma', 'max'], vary=disease_vary)

            return params, initial_conditions

    """Callback fucntion to store explored parameters during optimization. Function with shape from LMFIT documentation recommendations."""
    def callback_func(self, params, iter_, resid, init_conditions, tspan, data):
        if iter_ != -1:
            for name in self.dict_training_params.keys():
                self.dict_training_params[name].append(params[name].value)

    """"OBJECTIVE FUNCTION TO SOLVE"""
    def error_sectors(self, params, initial_conditions, tspan, data):
        if self.model=='SIR':
            idx = 1
        elif self.model=='SEIR':
            idx= 2

        #Solve our ODE with given params and initial conditions.
        sol = self.ode_solver(tspan, initial_conditions, params)

        #Init I cases
        initI = initial_conditions[idx-1]

        #Compute cumulated I cases to minimize (ODE returns daily cases not cumulated so we have to compute them.)
        sol_ = self.get_cumulated_cases(sol, self.last_I_cumul)

        if not self.fit_from_beginning: #Conditions for rolling window fit on fixed 40 days each time.
            sol_ =  sol_[self.train_window-40:self.train_window]

        #Return simple residual
        return ((sol_ - data)/1.0).ravel()

    """Function to compute cumulated cases of interest because ODE returns only daily cases."""
    def get_cumulated_cases(self, sol, initI, test_phase=False):
        #Compute cumulated I cases to minimize
        cumul_i = [initI]

        if self.model=='SIR': #Compute dSdt
            diff_i = np.diff(sol[:,0], 1)

        elif self.model=='SEIR':
            diff_S = np.diff(sol[:,0], 1)
            diff_E = np.diff(sol[:,1], 1)
            diff_i = diff_E - np.abs(diff_S)

        #If test phase : returns only the cumulated cases on the test time span
        if test_phase:
            diff_i = diff_i[self.train_window:self.train_window+self.test_window]
            #Add absolute value to previous cumulated values
            for i, diff_ in enumerate(diff_i):
                cumul_i.append(np.abs(diff_)+cumul_i[i])

        elif not test_phase:
            #Add absolute value to previous cumulated values
            for i, diff_ in enumerate(diff_i):
                cumul_i.append(np.abs(diff_)+cumul_i[i])

        return np.asarray(cumul_i[:])


    @staticmethod
    def compute_mse(errors):
        return np.mean(errors**2)

    @staticmethod
    def compute_mae(errors):
        return np.mean(np.abs(errors))

    def train_test_split(self, data, split_day):
        train_date_start = data.loc[split_day, 'date']
        train_date = data.loc[split_day+self.train_window, 'date']
        test_date = data.loc[split_day+self.train_window+self.test_window, 'date']

        train_df = data[data['date'] < train_date]
        train_df = train_df[train_date_start <= train_df['date']]

        test_df = data[data['date'] >= train_date]
        test_df = test_df[test_date > test_df['date']]

        return train_df, test_df

    @staticmethod
    def remove_artefacts(dataframe):
        """Avoid decreasing cumulated cases (artefacts in data computation or collection) """
        #Clean artefacts
        dataframe['dI'] = dataframe['I'].diff()
        #Init where daily cases are negative i.e decresing cumulated cases.
        dataframe.loc[0, 'dI'] = 0
        indexes = list(np.where(dataframe['dI']<0)[0])

        while len(indexes) > 0:
            for i in indexes: #Replace by previous value until there is no negative daily cases.
                dataframe.loc[i, 'I'] = dataframe.loc[i-1, 'I']

            #Update
            dataframe['dI'] = dataframe['I'].diff()
            indexes = list(np.where(dataframe['dI']<0)[0])

        return dataframe

    def data_processing(self, country):
        """Get data and remove artefacts"""
        try:
            EPI_data, country_attrs = country_covid(country, self.owid_file, model=self.model)
        except ValueError:
            print(f'incomplete data on {country}')
            return [], []

        df = EPI_data.drop(['N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R', 'D', 's'])
        df.R.fillna(0, inplace=True)
        df.ffill(axis=0, inplace=True)
        df.bfill(axis=0, inplace=True)
        thresh = self.cutoff_date

        if self.beta_step=='linear':
            self.stringency_country = get_stringency_data(df)
            self.mob_country = get_mobility_data(df, country, self.countries, self.iso_code2)
            #Save data
            save_to = './{0}/{1}/data_{2}_strin_mob.npy'.format(self.file_method, self.file_model, country)
            np.save(save_to, {'stringency': self.stringency_country , 'mobility' : self.mob_country})

        #Get data only for 2020
        df = df[df['date'] <= thresh]
        #Remove artefacts
        df = self.remove_artefacts(df)

        return df, country_attrs


    def prepare_training(self, df, country_attrs, disease_vary=True ):
        """Prepare initial conditions, parameters for training phase."""
        #Get ground truth : cumulated cases
        data = df.loc[0:, ['I']].values

        #Train-test split
        train_df, test_df = self.train_test_split(df, 0)
        #Keep data
        train_data, test_data = train_df['I'].values,  test_df['I'].values

        #Initialization
        initE, initI, initR = 1.0, 1.0, 0.0

        if self.beta_step=='constant':
            #Sub sample initial population for the model to work.
            max_cases = data[-1].item()
            initN = max_cases
        elif self.beta_step=='sigmoid' or self.beta_step=='linear':
            initN = country_attrs['population']

        #Initialiaze parameters and conditions
        params, initial_conditions = self.init_sectors(disease_vary=disease_vary, initN=initN, initE = initE, initI=initI, initR=initR)

        #If first training period, initialize with initI, else initialize with last value of cumulative cases
        self.last_I_cumul = initI

        #Saving explored params
        if self.beta_step=='sigmoid':
            self.dict_training_params = {i : [] for i in ['L', 'b']}
        elif self.beta_step=='linear':
            self.dict_training_params = {i : [] for i in ['w1', 'w2', 'b']}

        return train_data, test_data, params, initial_conditions


    def prepare_testing(self, df, result):
        #Get ground truth : cumulated cases
        data = df.loc[0:, ['I']].values
        self.last_I_cumul = data[self.train_window-1].item() #Init last cumulated cases number
        params_fitted = result.params
        return params_fitted

    def fit_test(self, df, country_attrs, disease_vary=True):
        #Prepare training
        train_data, test_data, params, initial_conditions = self.prepare_training(df, country_attrs, disease_vary=disease_vary)

        #TRAINING
        tspan_train = np.arange(1, 1+self.train_window, 1)
        mini = lmfit.Minimizer(self.error_sectors, params, fcn_args=(initial_conditions, tspan_train, train_data))
        result = mini.minimize( method='dual_annealing')
        ci = lmfit.conf_interval(mini, result)
        lmfit.printfuncs.report_ci(ci)
        #result = minimize(self.error_sectors, params, args=(initial_conditions, tspan_train, train_data), method='dual_annealing')
        #result = minimize(self.error_sectors, params, args=(initial_conditions, tspan_train, train_data), iter_cb = self.callback_func, method='dual_annealing')
        #result = minimize(self.error_sectors, params, args=(initial_conditions, tspan_train, train_data), method='leastsq', full_output = 1)
        #print(report_fit(result))

        #Store fit
        train_final = train_data + result.residual.reshape(train_data.shape)

        #Compute MAE and MSE for country
        MSE, MAE = self.compute_mse(result.residual.reshape(train_data.shape)), self.compute_mae(result.residual.reshape(train_data.shape))

        #Keep explored parameters
        #dict_training_params.append(self.dict_training_params)

        #TESTING
        params_fitted = self.prepare_testing(df, result)
        tspan_test = np.arange(1, self.train_window+self.test_window+1, 1)
        initial_conditions_test = initial_conditions
        #Predict with fitted parameters
        predicted = self.ode_solver(tspan_test, initial_conditions_test, params_fitted)

        #Compute cumulated I cases
        sol = self.get_cumulated_cases(predicted, self.last_I_cumul, test_phase=True)
        test_error = ((sol - test_data)/1.0).ravel()
        test_MSE, test_MAE = self.compute_mse(test_error), self.compute_mae(test_error)

        return ci, train_data, test_data, result, train_final, MSE, MAE, sol, test_MSE, test_MAE



    """"~~~~FITTING FUNCTIONS~~~~"""

    def fit_country(self, country, train_window, disease_vary=True, test_window = 15, fit_from_beginning=True):
        """Fit and test one country"""
        #try:
            #results = np.load('./{0}/{1}/TFmodel/result_{2}_{3}days.npy'.format(self.file_method, self.file_model, country, train_window), allow_pickle=True).item()
            #print('{0} model already fitted for {1} country on {2} days.'.format(self.model, country, train_window))

        #except FileNotFoundError:
        print(f'Fitting {self.model} for {country} on {train_window} days ...')

        self.fit_from_beginning = fit_from_beginning #Either extend the window for training or just slide a fixed size window.
        self.test_window = test_window
        self.train_window = train_window

        #Get preprocessed data for given country and model
        df, country_attrs = self.data_processing(country)

        #Fit and test
        ci, train_data, test_data, result, train_final, MSE, MAE, sol, test_MSE, test_MAE = self.fit_test(df, country_attrs, disease_vary)

        #Saving results
        #save_to_path = './{0}/{1}/result_{2}_{3}days.npy'.format(self.file_method, self.file_model, country, train_window)

        #np.save(save_to_path,  {'result': result, 'train_data': train_data,  'train_fit': train_final, 'train_metrics': {'mse': MSE, 'mae': MAE},
        #'test_data': test_data, 'test_predicted': sol, 'test_metrics': {'mse': test_MSE, 'mae': test_MAE}})
        return {'ci':ci, 'result': result, 'train_data': train_data,  'train_fit': train_final, 'train_metrics': {'mse': MSE, 'mae': MAE},
        'test_data': test_data, 'test_predicted': sol, 'test_metrics': {'mse': test_MSE, 'mae': test_MAE}}

    def rolling_window_fit(self, country, disease_vary=True, windows = [40, 80, 120, 160, 200], fit_from_beginning=True):
        """Rolling windows :
        ------------------------------"""
        self.fit_from_beginning = fit_from_beginning #Either extend the window for training or just slide a fixed size window.
        self.test_window = 15

        print(f'Fitting {country} data.')
        #Get preprocessed data for given country and model
        df, country_attrs = self.data_processing(country)

        #List of data to store for plots
        train = []
        test = []
        mse = []
        mae = []
        test_mse = []
        test_mae = []
        list_train_fitted= []
        list_result= []
        list_test_predicted= []
        dict_training_params = []

        #Loop over windows
        for i, split_day in enumerate(windows):
            print(f'...Training on rolling window {i+1}.')

            self.train_window = split_day

            #Fit and test
            train_data, test_data, result, train_final, MSE, MAE, sol, test_MSE, test_MAE = self.fit_test(df, country_attrs, disease_vary)

            #Store data
            train.append(train_data), test.append(test_data)
            list_train_fitted.append(train_final), list_result.append(result)
            mse.append(MSE), mae.append(MAE)
            list_test_predicted.append(sol)
            test_mse.append(test_MSE), test_mae.append(test_MAE)

            #Saving results
            save_to_path = './{0}/{1}/results_{2}_rolling_windows.npy'.format(self.file_method, self.file_model, country)

            try:
                dict_results = np.load(save_to_path, allow_pickle=True).item()
                dict_results['result'].append(result)
                dict_results['train_data'].append(train_data)
                dict_results['train_fit'].append(train_final)
                dict_results['train_metrics']['mse'].append(MSE)
                dict_results['train_metrics']['mae'].append(MAE)
                dict_results['test_data'].append(test_data)
                dict_results['test_predicted'].append(sol)
                dict_results['test_metrics']['mse'].append(test_MSE)
                dict_results['test_metrics']['mae'].append(test_MAE)

                np.save(save_to_path, dict_results)

            except FileNotFoundError:
                np.save(save_to_path,  {'result': list_result, 'train_data': train,  'train_fit': list_train_fitted, 'train_metrics': {'mse': mse, 'mae': mae},
                'test_data': test, 'test_predicted': list_test_predicted, 'test_metrics': {'mse': test_mse, 'mae': test_mae}, 'full_data' : df.loc[0:, ['I']].values})

    #Function to compute error bars
    @staticmethod
    def benchmark_error_bars(path_results):
        """
        """
        #get train residuals
        dict_results = np.load(path_results, allow_pickle=True).item()
        MSE = dict_results['train_metrics']['mse']

        ERROR = []

        for mse in MSE:
            ERROR_WINDOW_95 = []
            ERROR_WINDOW_90 = []
            ERROR_WINDOW_80 = []

            for h in range(1,16):
                #std_res = np.std(res.residual)
                #error_hat = std_res*np.sqrt(h)
                #error_hat = np.sqrt(h)
                ERROR_WINDOW_95.append(1.96*np.sqrt(mse*h))
                ERROR_WINDOW_90.append(1.64*np.sqrt(mse*h))
                ERROR_WINDOW_80.append(1.28*np.sqrt(mse*h))

            ERROR.append(np.stack((np.asarray(ERROR_WINDOW_95), np.asarray(ERROR_WINDOW_90), np.asarray(ERROR_WINDOW_80))))

        return ERROR
