from __future__ import division # always-float division
import numpy as np
import pandas as pd
import pprint
import os
import requests
from datetime import date
import glob
from zipfile import ZipFile

# Easy interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Interactive plots in notebook
from IPython.display import HTML, Image, display
from ipywidgets.widgets import interact, IntSlider, FloatSlider, Layout, ToggleButton, ToggleButtons, fixed, Checkbox

# ISO codes for OWID data
import pycountry


##DATA

def download_csv(url, path, prefix):
    """
    Downloads a CSV from 'url', saves it to 'path' folder with filename 'prefix'_DD-MM-YYYY formatted at today's date
    """
    response = requests.get(url, allow_redirects=True)
    today = str(date.today())
    filepath = f'{path}/{prefix}_{today}.csv'
    open(filepath, 'wb').write(response.content)
    return filepath

def update_owid(path):
    """
    Updates Our World In Data database and saves it to 'path' folder. Renames it to owid_DD-MM-YYYY with today's date
    """
    filepath = download_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv', path, 'owid')
    print(f'Downloaded Our World In Data Coronavirus data to \n\t{filepath}')
    return filepath

def update_hospitalieres(path):
    """
    Updates Santé Publique France - Données Hospitalières database and saves it to 'path' folder. Renames it to hospitalieres_DD-MM-YYYY with today's date
    """
    filepath = download_csv('https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7', path, 'hospitalieres')
    print(f'Downloaded Données hospitalières Santé Publique data to \n\t{filepath}')
    return filepath

def countries_owid(file):
    """
    Return list of countries in an OWID database from 'file'
    """
    data = pd.read_csv(file, sep=",",header=0)
    return data['iso_code'].unique()

def extract_owid(file, country_code='FRA'):
    """
    Extracts 'country_code' data from OWID database (date, I, D, s), creates country attributes from single-valued columns
    """
    # Open, country-filter & relevant column-filer data
    data = pd.read_csv(file, sep=",",header=0)
    countries = data['iso_code'].unique()
    if country_code not in countries:
        country_code = 'FRA'
        print('Invalid country code, extracting France by default')
    data = data[data['iso_code'] == country_code]
    cols = ['date', 'total_cases', 'total_deaths', 'stringency_index', 'population']
    # Drop unnecesary columns
    data = data.drop([c for c in data.columns if c not in cols], axis=1)
    # Get single-valued columns to return as country attribute
    uniques = unique_columns(data)
    attrs = {}
    for col in uniques:
        attrs[col] = data[col].iloc[0]
    # Drop single-valued columns
    data = data.drop(uniques, axis=1)
    data.reset_index(drop=True, inplace=True)
    return data, attrs

def hospitalieres_summary(file):
    """
    Creates national summary database from daily hospitary data per department, saved in 'file'
    """
    # Open CSV & create df with present dates
    data = pd.read_csv(file, sep=";",header=0)
    new_df = pd.DataFrame()
    dates= data['jour'].unique()
    new_df['date'] = dates
    # Filter data: separate male & female
    data = data[data['sexe'] != 0]
    # For relevant columns (hospitalized, dead, sent to home, in reanimation) sum for all departments in a given day
    for c in ['hosp', 'rea', 'rad']:
        cum_data = data.groupby('jour')[c].sum().to_numpy()
        # Assign relevant summarized column to storage df
        new_df[c] = cum_data
    return new_df

def datefy(s):
    """
    returns YYYY-MM-DD from YYYY-MM-DD or DD/MM/YYYY
    """
    l2 = s.split('/')
    if len(s.split('-')) == 3:
        return s
    elif len(l2) == 3:
        return '-'.join(reversed(l2))
    else:
        print('Non consistent date format')
        return 'nan'


def country_covid(country, owid_file, model='SIR'):
    """
    Extracts 'country' time series from OWID database saved in 'owid_file'. Can add hospital data if 'country'==France and 'hosp_file' is specified.
    Time series starts when infected are positive
    """
    # Get ISO-3 code for country to later use as index for OWID database
    try:
        code = pycountry.countries.search_fuzzy(country)[0].alpha_3
    except LookupError:
        print(f'{country} not found in country dictionary.')
        return
    covid_data, country_attrs = extract_owid(owid_file, country_code=code)

    if model=='SIR'or model=='SEIR':
        covid_data = covid_data.sort_values(by='date')
        covid_data = covid_data.reset_index(drop=True)
        # Oldest EPI values are all 0 (I, R, D)
        covid_data.loc[0, covid_data.columns != 'date'] = covid_data.loc[0, covid_data.columns != 'date'].apply(lambda x: 0)
        # Forward-fill NaN: old value is maintained until not-NaN value
        covid_data.ffill(axis=0, inplace=True)
        # Rename columns
        covid_data.columns = ['date', 'I', 'D', 's']
        # Compute S
        #covid_data['S'] = country_attrs['population'] - covid_data['I'] - covid_data['D']
        covid_data['S'] = country_attrs['population'] - covid_data['I']
        covid_data = covid_data[covid_data['I'] > 0]
        covid_data.reset_index(drop=True, inplace=True)
        covid_data['N_effective'] = country_attrs['population'] - covid_data['D']
        covid_data.bfill(axis=0, inplace=True)

        # France case with hospital data
        """elif code == 'FRA' and hosp_file != '':
            hosp_database = hospitalieres_summary(hosp_file)
            # Fix date notation
            hosp_database.date = hosp_database.date.apply(lambda s: datefy(s))
            # Mean date duplicates
            hosp_database = hosp_database.groupby(by='date', as_index=False).mean()
            covid_data = hosp_database.merge(covid_data, on='date', how='outer')
            covid_data = covid_data.sort_values(by='date')
            covid_data = covid_data.reset_index(drop=True)
            # Oldest EPI values are all 0 (I, R, D)
            covid_data.loc[0, covid_data.columns != 'date'] = covid_data.loc[0, covid_data.columns != 'date'].apply(lambda x: 0)
            # Forward-fill NaN: old value is maintained until not-NaN value
            covid_data.ffill(axis=0, inplace=True)
            # Rename columns
            covid_data.columns = ['date', 'H', 'Reanimation', 'R', 'I', 'D', 's']
            # Compute S
            covid_data['S'] = country_attrs['population'] - covid_data['I'] - covid_data['D'] - covid_data['H'] - covid_data['Reanimation']"""
        # general country
    else:
        covid_data = covid_data.sort_values(by='date')
        covid_data = covid_data.reset_index(drop=True)
        # Oldest EPI values are all 0 (I, R, D)
        covid_data.loc[0, covid_data.columns != 'date'] = covid_data.loc[0, covid_data.columns != 'date'].apply(lambda x: 0)
        # Forward-fill NaN: old value is maintained until not-NaN value
        covid_data.ffill(axis=0, inplace=True)
        # Rename columns
        covid_data.columns = ['date', 'I', 'D', 's']
        # Compute S
        covid_data['S'] = country_attrs['population'] - covid_data['I'] - covid_data['D']
        covid_data = covid_data[covid_data['I'] > 0]
        covid_data.reset_index(drop=True, inplace=True)
        covid_data['N_effective'] = country_attrs['population'] - covid_data['D']
    #     covid_data['beta'] = -covid_data['N_effective'] * covid_data['S'].diff() / (covid_data['I'] * covid_data['S'])
    #     covid_data['mu'] = covid_data['D'].diff() / covid_data['I']
        covid_data.bfill(axis=0, inplace=True)

    return covid_data, country_attrs

def get_oxford_data(country):
    path = 'OXCRGT_2020_data.csv'

    #Variables to keep from Oxford dataset
    IP_COLUMNS = ['C1_School closing',
                  'C2_Workplace closing',
                  'C3_Cancel public events',
                  'C4_Restrictions on gatherings',
                  'C5_Close public transport',
                  'C6_Stay at home requirements',
                  'C7_Restrictions on internal movement',
                  'C8_International travel controls',
                  'H1_Public information campaigns',
                  'H2_Testing policy',
                  'H3_Contact tracing',
                  'H6_Facial Coverings']

    #Get dataset
    df = pd.read_csv(path,
         parse_dates=['date'],
         encoding="ISO-8859-1",
         dtype={"country": str,
                "iso_code": str},
         error_bad_lines=False)

    #KEEP_COLUMNS = ["CountryName", "CountryCode", "RegionName", "Date", "ConfirmedCases"]
    #Drop Brazil regions
    #df.drop(labels = df[df.CountryName=='Brazil'].index[0:-1], axis=0, inplace=True)
    #df = df.reset_index(0, drop=True)
    return df[df['country'] == country].reset_index(0, drop=True)


def unique_columns(df):
    """
    Return name of columns with an unique value
    """
    return [c for c in df.columns if len(df[c].unique()) == 1]

def save_res_local(res=None):
    #Dict of parameters and optimal values after fitting
    r = {'param':list(res.params.valuesdict()) , 'value': [res.params[p].value for p in list(res.params.valuesdict())]}
    #Create dataframe
    df_result= pd.DataFrame(data=r)
    #Save to csv
    df_result.to_csv('results_fitted_france.csv')

# get mobility
def get_mobility(path):
    mobility_df = pd.read_csv(path)
    #Drop non useful information
    mobility_df.drop(['sub_region_1','sub_region_2', 'metro_area', 'iso_3166_2_code','census_fips_code', 'place_id'], axis=1, inplace=True)
    #Rename column of countries
    mobility_df = mobility_df.rename(columns={'country_region': 'CountryName'})
    #Drop duplicates in dates
    mobility_df = mobility_df.drop_duplicates(subset=['CountryName','date'], keep='first', ignore_index=True)
    return mobility_df

def get_data_w_mobility(iso_code2='None'):
    MOBILITY_COLUMNS=['retail_and_recreation_percent_change_from_baseline',
            'grocery_and_pharmacy_percent_change_from_baseline',
            'parks_percent_change_from_baseline',
            'transit_stations_percent_change_from_baseline',
            'workplaces_percent_change_from_baseline',
            'residential_percent_change_from_baseline']

    zip_ = ZipFile('Region_Mobility_Report_CSVs.zip', 'r')
    file_to_extract1 = '2020_{}_Region_Mobility_Report.csv'.format(iso_code2)
    file_to_extract2 = '2021_{}_Region_Mobility_Report.csv'.format(iso_code2)
    zip_.extract(file_to_extract1, path='./temp_mobility')
    zip_.extract(file_to_extract2, path='./temp_mobility')

    mobility_df1 = get_mobility('./temp_mobility/'+file_to_extract1)
    mobility_df2 = get_mobility('./temp_mobility/'+file_to_extract2)

    mobility_df = pd.concat([mobility_df1, mobility_df2], ignore_index=True)
    mob_data =  mobility_df[MOBILITY_COLUMNS].values.sum(axis=1)
    #Normalize between 0 and 1
    mob_data_norm = (mob_data - mob_data.min())/(mob_data.max() - mob_data.min())
    #mob_data_norm = 2*(mob_data - mob_data.min())/(mob_data.max() - mob_data.min()) -1

    mobility_df['mobility_index'] = mob_data_norm

    mobility_df.drop(MOBILITY_COLUMNS+['country_region_code', 'CountryName'], axis=1, inplace=True)

    #remove temporary files
    os.remove('./temp_mobility/'+file_to_extract1)
    os.remove('./temp_mobility/'+file_to_extract2)

    return mobility_df

"""Get normalzed mobility data and smooth it"""
def get_mobility_data(df, country, list_countries, list_iso_codes):
    #Add mobility data
    k = np.where(np.asarray(list_countries) == country)[0].item()
    mobility_df = get_data_w_mobility(list_iso_codes[k])
    #Merge datasets
    df = df.merge(mobility_df[['mobility_index','date']], left_on=['date'], right_on=['date'], how='left')
    df['mobility_index'].bfill(inplace=True) #fill nan
    df['mobility_index_smooth']= df['mobility_index'].rolling(20, center=False).mean().reset_index(0, drop=True).fillna(0.5) # 0.5 means no difference in mobility from previous year
    return df.loc[0:, ['mobility_index_smooth']].values.ravel()

"""Get stringency data, normalize it and smooth it ."""
def get_stringency_data(df):
    #Stringency
    data_s = df.loc[0:, ['s']].values
    data_s = data_s.reshape(len(data_s))
    data_s_norm = (data_s - data_s.min())/(data_s.max() - data_s.min())
    #Smooth stringency
    df['s_norm'] = data_s_norm
    df['stringency_smooth']= df['s_norm'].rolling(20, center=False).mean().reset_index(0, drop=True).fillna(0.001)
    s_smooth = df.loc[0:, ['stringency_smooth']].values
    return s_smooth


def get_NPI_data(df):
    """Get stringency data, normalize it and smooth it ."""

    IP_COLUMNS = ['C1_School closing',
                  'C2_Workplace closing',
                  'C3_Cancel public events',
                  'C4_Restrictions on gatherings',
                  'C5_Close public transport',
                  'C6_Stay at home requirements',
                  'C7_Restrictions on internal movement',
                  'C8_International travel controls',
                  'H1_Public information campaigns',
                  'H2_Testing policy',
                  'H3_Contact tracing',
                  'H6_Facial Coverings']

    #NPIs
    npi_data = []
    for npi in IP_COLUMNS:
        data_s = df.loc[0:, [npi]].values
        data_s = data_s.reshape(len(data_s))
        data_s_norm = (data_s - data_s.min())/(data_s.max() - data_s.min())
        #Smooth stringency
        df['s_norm'] = data_s_norm
        df['stringency_smooth'] = df['s_norm'].rolling(20, center=False).mean().reset_index(0, drop=True).fillna(0.001)
        s_smooth = df.loc[0:, ['stringency_smooth']].values
        npi_data.append(s_smooth)

    return npi_data


#"""Turn stringency into continuous value for integration"""
#def get_strin_continuous(t, stringency_country):
    #if t >= len(stringency_country)-1:
    #    return stringency_country[-1]
    #else:
        #return stringency_country[int(np.floor(t))] + (t - np.floor(t))*(stringency_country[int(np.ceil(t))]-stringency_country[int(np.floor(t))])

"""Turn discrete data into continuous value for integration"""
def get_continuous(t, data):
    if t >= len(data)-1:
        return data[-1]
    else:
        return data[int(np.floor(t))] + (t - np.floor(t))*(data[int(np.ceil(t))]-data[int(np.floor(t))])

"""Define if model is overestimating or underestimating"""
def compute_fraction_mae(data, preds):
    assert len(data) == len(preds), 'Inputs must have same length.'
    E_pos = []
    E_neg = []

    frac_under =0
    frac_over=0

    for true, pred in zip(data, preds):
        x = true - pred
        #print(x)
        if x > 0:
            #underestimating
            frac_under +=1
            E_pos.append(abs(x))
        elif x <= 0:
            #overestimating
            frac_over +=1
            E_neg.append(abs(x))

    mae_underestimate = sum(E_pos)/len(data)
    mae_overestimate = sum(E_neg)/len(data)

    return frac_over/len(data), mae_overestimate, frac_under/len(data), mae_underestimate


## PLOTS
def plot_dualaxis(x, y1, y2, namesy1, namesy2, title='', log_y2=False, static=False, xtitle='Day'):
    """
    Plots in plotly on dual y axis with a common 'x' series (Days). 'namesy1' and 'namesy2' specify variable names for series in 'y1' and 'y2' respectively.
    Last item in 'namesy1' and 'namesy2' is title of the axis (len is 1-bigger than y1, y2 respectively)
    'log_y2' flag for log scale on second axis, 'title' is plot title and 'static' is flag for png or interactive js plotly
    """
    if type(y1) != list:
        y1 = [y1]
    if type(y2) != list:
        y2 = [y2]
    if type(namesy1) != list:
        namesy1 = [namesy1, namesy1]
    if type(namesy2) != list:
        namesy2 = [namesy2, namesy2]
    if type(x) == int:
        tspan = np.arange(0, x, 1)
    else:
        tspan = x

    scale = 'linear'
    if log_y2:
        scale = 'log'

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for yi in range(len(y1)):
        fig.add_trace(go.Scatter(x=tspan, y=y1[yi], mode='lines+markers', name=namesy1[yi]))
    for yj in range(len(y2)):
        fig.add_trace(go.Scatter(x=tspan, y=y2[yj], mode='lines+markers', name=namesy2[yj]), secondary_y=True)

    fig.update_layout(title=title,
                           xaxis_title=xtitle,
                           yaxis_title=namesy1[-1],
                           yaxis2_title=namesy2[-1],
                           yaxis2_type=scale,
                           title_x=0.5,
                          width=1000, height=600
                         )

    if static:
        img_bytes = fig.to_image(format="png")
        display(Image(img_bytes))
    else:
        fig.show()

def simulate_plot(result, days, initial_conditions, static=False):
    """
    Simulation plot for 'result' fitted model parametes, for 'days' timespan and 'initial_conditions'
    """
    params = result.params
    tspan = tspan = np.arange(0, days, 1)
    n_sectors = len(initial_conditions)
    sol = np.zeros((days, 3))
    for i in range(n_sectors):
        sol_i = ode_solver(tspan, initial_conditions[i], params, i)
        sol += sol_i[:, 2:5]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tspan, y=sol[:, 0], mode='lines+markers', name='Infections'))
    fig.add_trace(go.Scatter(x=tspan, y=sol[:, 1], mode='lines+markers', name='Recovered'))
    fig.add_trace(go.Scatter(x=tspan, y=sol[:, 2], mode='lines+markers', name='Deaths'))
    fig.update_layout(title='SEIRD per sector: Simulation',
                           xaxis_title='Day',
                           yaxis_title='Counts',
                           title_x=0.5,
                          width=1000, height=600
                         )
    if STATIC_PLOTS:
        img_bytes = fig.to_image(format="png")
        display(Image(img_bytes))
    else:
        fig.show()
