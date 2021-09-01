import numpy as np
import os
import matplotlib.pyplot as plt
import random
from error_predictor import ErrorPredictor
from epi_model import EPI_Model
from utils import *
import argparse

#Required arguments when calling script
parser = argparse.ArgumentParser()

parser.add_argument("-data_type", "--data_type",
                        dest="data_type",
                        type=str,
                        required=True,
                        help="Specify if bootstrap method should be applied on 'countries' data or 'simulations' data.")
args = parser.parse_args()

DATA_TYPE = args.data_type.lower()


#FUNCTIONS TO GENERATE ALTERNATIVE DATA
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_alternative_samples(data, max_days=360, skip_days=15):
    samples = []
    Ys=[]

    for T in range(4):
        rand_points = [np.random.randint(i-skip_days,i) for i in [np.arange(skip_days, max_days, skip_days)]]
        rand_values = data[rand_points[0]]

        Y = data.copy()
        prev_idx=0
        for i, idx in enumerate(rand_points[0]):
            if idx != 0:
                Y[prev_idx:idx] = rand_values[i]

            if i==max_days/skip_days and idx !=max_days:
                Y[idx:max_days] = data[-1].item()

            prev_idx = idx


        samples.append(smooth(np.pad(Y, int(skip_days), mode='constant') , skip_days+10)[:max_days])

    return samples

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DATA_TYPE=='simulations':

    #Load
    FILE_METHOD = 'results_error_predictor'
    FILE_MODEL = 'SIR'
    simulations_dict = np.load('./{0}/{1}/simulations_{2}samples.npy'.format(FILE_METHOD, FILE_MODEL, 300), allow_pickle=True).item()

    simulations = simulations_dict['simulations']
    beta_simulations = simulations_dict['betas']
    cases_simulations= simulations_dict['cases']
    populations = simulations_dict['populations']

    #Generate 4 alternative samples for each simulation
    samples = [get_alternative_samples(cases) for cases in cases_simulations]

    #Fit SIR model
    Predictor = ErrorPredictor(model='SIR')
    #Path
    save_to_path = './{0}/{1}/results_simulations_addit_samples.npy'.format(FILE_METHOD, FILE_MODEL)
    #Loop
    for i in range(100):
        cases = np.concatenate((cases_simulations[i].reshape(1, 360), np.stack(samples[i])))
        stringency = np.repeat(simulations[i].reshape(1,len(simulations[i])), 5, axis=0)
        population = np.repeat(populations[i], 5, axis=0)
        beta_simu = np.repeat(beta_simulations[i].reshape(1,len(beta_simulations[i])), 5, axis=0)

        Predictor.fit(cases, stringency, population, beta_simu, train_window=100, samples=5, SAVE_TO_PATH=save_to_path)

    #Get final results
    results_simu = np.load('./results_error_predictor/SIR/results_simulations_addit_samples.npy', allow_pickle=True)
    #Get errors
    PREDICTED_ERROR_BARS=[]
    SIMULATIONS = []

    for i in range(0, len(results_simu), 5):
        #Save simulation (100 in total)
        SIMULATIONS.append(results_simu[i])

        #Get error bars from alternative data
        error_bars=[]
        for ALT in range(1,5):
            error_bars.append(np.abs(results_simu[i+ALT]['test_predicted'] - results_simu[i+ALT]['test_data']))

        #Get standard deviation
        predicted_error_bar = np.std(error_bars, axis= 0)

        PREDICTED_ERROR_BARS.append(predicted_error_bar)

    #Save results
    np.save('./results_error_predictor/SIR/alternative_errorbars_simulations.npy', {'results_simulations' : SIMULATIONS, 'error_bars': PREDICTED_ERROR_BARS})


%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DATA_TYPE=='countries':

    #Fitting SIR model on alternative data of each country
    for COUNTRY, SCALE in zip(['United Kingdom', 'United States','Australia', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'India',
        'Israel', 'Italy', 'Japan', 'Luxembourg', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Russia', 'South Africa', 'Korea, Republic of', 'Spain', 'Sweden',
        'Taiwan'], [1.2, 1.2, 1., 1., 1.1, 1.1, 1.3, 1.1 , 1.2 , 1.05,1.05,1.05, 1.5, 1.05, 1.1, 1.2, 1.05, 1.05, 1.05, 1.2, 1.1, 1.5,1.5,
                    1., 1.02, 1.1,1.02]):
        SAVE_TO_PATH = './results_stringency/SIR/result_{0}_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW)

        #Get stringency data and population to fit alternative real data
        try:
            res = np.load(SAVE_TO_PATH, allow_pickle=True).item()
        except FileNotFoundError:
            print("Make sure to fit the SIR model on all countries before applying bootstrap method. Run 'countries_forecast.py' script.")

        population = res['population']
        stringency_data = res['stringency_data']
        train_data = res['train_data']
        test_data = res['test_data']

        #Create alternative data
        full_data = np.concatenate((train_data, test_data), axis=0)
        samples = get_alternative_samples(full_data, max_days=115, skip_days=10)
        samples = np.asarray(samples)*SCALE

        print(f"Fitting alternative data for {COUNTRY}...")
        #Create dataframe
        results_dict = []
        for alt_data in samples:

            #Extend full data by one day for functions compatibility
            alt_data = np.concatenate((alt_data, np.zeros(2).reshape(2,)))

            df = pd.DataFrame(data={'date' : pd.date_range("2020", freq="D", periods=len(alt_data)), 'I': alt_data})
            country_attrs = {'population' : population}

            model.stringency_country = stringency_data

            #Prepare training with fixed gamma (disease_vary=False)
            train_data, test_data, params, initial_conditions = model.prepare_training(df, country_attrs, disease_vary=False)

            #Training and forecast
            tspan_train = np.arange(1, 1+100, 1)
            ci, train_data, test_data, result, train_final, MSE, MAE, sol, test_MSE, test_MAE = model.fit_test(df, country_attrs, disease_vary=False)

            #Save dictionary of results for each simulation
            results_dict.append({ 'ci' :ci, 'result': result, 'train_data': train_data,  'stringency_data' : stringency_data,  'train_fit': train_final,
            'test_data': test_data, 'test_predicted': sol, 'population':population})

        #Save results for alternative data
        np.save('./results_stringency/SIR/result_{0}_alternatives_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW), results_dict)

    #Repeat with additional countries and respective scales
    for COUNTRY, SCALE in zip(['Chile', 'Colombia', 'China', 'Egypt', 'Congo', 'Guana', 'Iran', 'Libya', 'Kenya', 'Mexico', 'New Zealand',
               'Qatar', 'Peru', 'Sudan', 'Turkey', 'Bangladesh', 'Somalia', 'Singapore', 'Senegal','Ukraine', 'Vietnam',
               'Yemen', 'Oman', 'Uruguay', 'Venezuela', 'United Arab Emirates', 'Pakistan'], [1.2, 1.2, 1.05, 1.2, 1.3, 1.1, 1.15, 1.1 , 1.2 , 1.05,1.05,1.05, 1.2, 1.05, 1.1, 1.2, 1.05, 1., 1.05, 1.0, 1.1, 1.1,1.5,
                    1., 1.2, 1.1,1.3]):
        SAVE_TO_PATH = './results_stringency/SIR/result_{0}_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW)

        #Get stringency data and population to fit alternative real data
        res = np.load(SAVE_TO_PATH, allow_pickle=True).item()
        population = res['population']
        stringency_data = res['stringency_data']
        train_data = res['train_data']
        test_data = res['test_data']

        #Create alternative data
        full_data = np.concatenate((train_data, test_data), axis=0)
        samples = get_alternative_samples(full_data, max_days=115, skip_days=10)
        samples = np.asarray(samples)*SCALE

        print(f"Fitting alternative data for {COUNTRY}...")
        #Create dataframe
        results_dict = []
        for alt_data in samples:
            #Extend full data by one day for functions compatibility
            alt_data = np.concatenate((alt_data, np.zeros(2).reshape(2,)))
            df = pd.DataFrame(data={'date' : pd.date_range("2020", freq="D", periods=len(alt_data)), 'I': alt_data})
            country_attrs = {'population' : population}

            model.stringency_country = stringency_data

            #Prepare training with fixed gamma (disease_vary=False)
            train_data, test_data, params, initial_conditions = model.prepare_training(df, country_attrs, disease_vary=False)

            #Training and forecast
            tspan_train = np.arange(1, 1+100, 1)
            ci, train_data, test_data, result, train_final, MSE, MAE, sol, test_MSE, test_MAE = model.fit_test(df, country_attrs, disease_vary=False)

            #Save dictionary of results for each simulation
            results_dict.append({ 'ci' :ci, 'result': result, 'train_data': train_data,  'stringency_data' : stringency_data,  'train_fit': train_final,
            'test_data': test_data, 'test_predicted': sol, 'population':population})

        #Save results for alternative data
        np.save('./results_stringency/SIR/result_{0}_alternatives_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW), results_dict)



        #Get final results
        #Create dict of countries with respective error bars from alternative data
        PREDICTED_ERROR_BARS = dict()

        for COUNTRY in ['United Kingdom', 'United States','Australia', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'India',
                'Israel', 'Italy', 'Japan', 'Luxembourg', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Russia', 'South Africa', 'Korea, Republic of', 'Spain', 'Sweden',
                'Taiwan']:

            #Get real data results
            real_results = np.load('./results_stringency/SIR/result_{0}_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW), allow_pickle=True).item()
            #Get alternative data results
            results_dict = np.load('./results_stringency/SIR/result_{0}_alternatives_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW), allow_pickle=True)

            #Get alternative data error bars
            error_bars=[]
            for i in range(len(results_dict)):
                error_bars.append(np.abs(results_dict[i]['test_predicted'] - results_dict[i]['test_data']))

            #Get standard deviation
            predicted_error_bar = np.std(error_bars, axis= 0)
            PREDICTED_ERROR_BARS[COUNTRY] = predicted_error_bar

        #Save dict
        np.save('./results_stringency/SIR/alternative_errorbars_countries.npy', PREDICTED_ERROR_BARS)

        #Create dict of countries with respective error bars from alternative data for new countries
        PREDICTED_ERROR_BARS = dict()

        for COUNTRY in ['Chile', 'Colombia', 'China', 'Egypt', 'Congo', 'Guana', 'Iran', 'Libya', 'Kenya', 'Mexico', 'New Zealand',
                   'Qatar', 'Peru', 'Sudan', 'Turkey', 'Bangladesh', 'Somalia', 'Singapore', 'Senegal','Ukraine', 'Vietnam',
                   'Yemen', 'Oman', 'Uruguay', 'Venezuela', 'United Arab Emirates', 'Pakistan']:

            #Get real data results
            real_results = np.load('./results_stringency/SIR/result_{0}_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW), allow_pickle=True).item()
            #Get alternative data results
            results_dict = np.load('./results_stringency/SIR/result_{0}_alternatives_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW), allow_pickle=True)

            #Get alternative data error bars
            error_bars=[]
            for i in range(len(results_dict)):
                error_bars.append(np.abs(results_dict[i]['test_predicted'] - results_dict[i]['test_data']))

            #Get standard deviation
            predicted_error_bar = np.std(error_bars, axis= 0)
            PREDICTED_ERROR_BARS[COUNTRY] = predicted_error_bar

        #Save dict
        np.save('./results_stringency/SIR/alternative_errorbars_newcountries.npy', PREDICTED_ERROR_BARS)
