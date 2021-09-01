import numpy as np
import os
import matplotlib.pyplot as plt
from epi_model import EPI_Model
from utils import *

# Relative path to EPI data folder
DATA_PATH = './data/clean/EPI'
# Download OWID data
existing = glob.glob(DATA_PATH+"/*"+str(date.today())+".csv")
owid_file = update_owid(DATA_PATH)
#Parameters file
params_sir_countries_lin = './params_new/params_countries/params_SIR_countries_linear_stringency.csv'
#Define SIR model
model = EPI_Model(model='SIR', beta_step='stringency',
                      PARAMS_FILE=params_sir_countries_lin, OWID_FILE=owid_file)
#Set training window
TRAIN_WINDOW = 100

#Fitting SIR on all countries
for COUNTRY in ['United Kingdom', 'United States','Australia', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'India',
        'Israel', 'Italy', 'Japan', 'Luxembourg', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Russia', 'South Africa', 'Korea, Republic of', 'Spain', 'Sweden',
        'Taiwan', 'Chile', 'Colombia', 'China', 'Egypt', 'Congo', 'Guana', 'Iran', 'Libya', 'Kenya', 'Mexico', 'New Zealand',
           'Qatar', 'Peru', 'Sudan', 'Turkey', 'Bangladesh', 'Somalia', 'Singapore', 'Senegal','Ukraine', 'Vietnam',
           'Yemen', 'Oman', 'Uruguay', 'Venezuela', 'United Arab Emirates', 'Pakistan']:
    SAVE_TO_PATH = './results_stringency/SIR/result_{0}_{1}days.npy'.format(COUNTRY, TRAIN_WINDOW)
    #Fit model on 100 days and predict 15 next days
    print(f'Fitting {COUNTRY}...')
    model.fit_country(COUNTRY, train_window=100, disease_vary=False, fit_from_beginning=True, SAVE_TO_PATH=SAVE_TO_PATH)
