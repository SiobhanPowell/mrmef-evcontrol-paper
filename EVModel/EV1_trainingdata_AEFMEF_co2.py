import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
import os

from EV1_shared_functions import preprocess
from EV1_shared_functions import end_times_and_load
from EV1_shared_functions import LoadModel

        
# Load files
df_home = pd.read_csv('Data/simulated_home_sessions_weekday_20230303.csv', index_col=0)
df_home = preprocess(df_home)
df_work = pd.read_csv('Data/simulated_work_sessions_weekday_20230303.csv', index_col=0)
df_work = preprocess(df_work)


date = '20240208'
print('---'*5, 'AEF co2', '---'*5)
signals_df_aefmef = pd.read_csv('Data/signals_df_aef_mef_storagebefore_noevbase_co2price_01dolperkg_20240208.csv', index_col=0)
folder = 'Outputs_EVControl_aefmef_co2_'+date+'/'
if not os.path.isdir(folder):
    os.mkdir(folder)
    
cols_to_optimize = [control+'_'+weekday+'_'+str(year) for control in ['AEF', 'MEF'] for weekday in ['weekday', 'weekend'] for year in [2023, 2028, 2033]]

cols1 = cols_to_optimize
cols2 = []
all_cols = cols_to_optimize

num_runs = 100 
num_cars = 200
save_date = '20240208'

num_time_steps = 96
time_steps_per_hour = 4


inds_all = {'home':np.zeros((num_runs, num_cars)), 'work':np.zeros((num_runs, num_cars))}
input_data = {'home':np.zeros((num_runs, num_time_steps)), 'work':np.zeros((num_runs, num_time_steps))}
output_data = {'home':{col:np.zeros((num_runs, num_time_steps)) for col in all_cols}, 'work':{col:np.zeros((num_runs, num_time_steps)) for col in all_cols}}

tic = time.time()

for i in range(num_runs):
    if np.mod(i, 5) == 0:
        toc = time.time()
        print('Done ', i, 'runs')
        print('Time since last mark: ', toc-tic)
        
        for name in ['home', 'work']:
            np.save(folder+'control_input_data_uncontrolled_'+name+'_'+save_date+'.npy', input_data[name])
            for col in all_cols:
                np.save(folder+'control_output_data_'+col+'_'+name+'_'+save_date+'_nonreg.npy', output_data[name][col])
            np.save(folder+'control_indices_'+name+'_'+save_date+'.npy', inds_all[name])

        tic = time.time()
    
    
    for name, df in {'home':df_home, 'work':df_work}.items():

        inds = np.random.choice(range(df.shape[0]), num_cars)
        inds_all[name][i, :] = inds
        start_times = (df.loc[inds, 'start_15min'].values).astype(int)
        durations = (df.loc[inds, 'Session Time (15min)'].values).astype(int)
        energies = df.loc[inds, 'Energy (kWh)'].values
        end_times, uncontrolled_load = end_times_and_load(start_times, energies, durations, 6.6)

        load = LoadModel(num_sessions=num_cars)
        load.input_data(uncontrolled_load, start_times, end_times, energies)
        input_data[name][i, :] = uncontrolled_load
        
        for col in cols1:
            obj = np.copy(np.repeat(signals_df_aefmef[col].values, time_steps_per_hour))
            load.aef_controlled_load(obj)
            output_data[name][col][i, :] = np.sum(load.solar_controlled_power, axis=1)

toc = time.time()
print('Done ', i, 'runs')
print('Time since last mark: ', toc-tic)

for name in ['home', 'work']:
    np.save(folder+'control_input_data_uncontrolled_'+name+'_'+save_date+'.npy', input_data[name])
    for col in all_cols:
        np.save(folder+'control_output_data_'+col+'_'+name+'_'+save_date+'_nonreg.npy', output_data[name][col])
    np.save(folder+'control_indices_'+name+'_'+save_date+'.npy', inds_all[name])

