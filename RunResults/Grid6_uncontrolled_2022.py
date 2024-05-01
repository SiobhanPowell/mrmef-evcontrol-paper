import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
import scipy
import copy
from simple_dispatch import bidStack
from future_grid import FutureDemand_v2
from future_grid import FutureGrid_v2
import cvxpy
import os

gd = pickle.load(open('../Data/generator_data_short_WECC_2022_cluster.obj', 'rb'))
years = np.arange(2023, 2038)

date = '20240212'
folder2 = 'Results_AEFMEF_2022BASE_'+date
if not os.path.isdir(folder2):
    os.mkdir(folder2)
    
scaling = pd.read_csv('../Data/scaling_generation_fractions_2024final.csv', index_col=0)
max_rates = {}
caps = {}
for i in scaling.index:
    max_rates[int(scaling.loc[i, 'Year'])] = scaling.loc[i, 'Battery MW']
    caps[int(scaling.loc[i, 'Year'])] = 4 * scaling.loc[i, 'Battery MW']

time_array = np.arange(52)+1

# Reference
for year in years:
    save_str = folder2+'/noblock_year'+str(year)+'_storagebefore'
    print(save_str)
    
    demand1 = FutureDemand_v2(gd, year=year, base_year=2022)
    demand1.set_up_ready(evs=False, block=False)
    demand1.update_total()

    grid1 = FutureGrid_v2(gd, base_year=2022)
    grid1.add_generators(year)
    grid1.drop_generators(year)
    grid1.year = year
    grid1.future = demand1
    
    grid1.run_storage_before_capacitydispatch(caps[year], max_rates[year], allow_negative=True)
    grid1.future.demand.demand = np.copy(grid1.storage_before.df.comb_demand_after_storage.values)
    grid1.run_dispatch(save_str, verbose=False, time_array=time_array, result_date=date, coal_downtime=False, year=year)
    
    save_str = folder2+'/flat_block_year'+str(year)+'_storagebefore'
    print(save_str)

    demand1 = FutureDemand_v2(gd, year=year, base_year=2022)
    demand1.set_up_ready(evs=False, verbose=False, block=True, hourly_signal=None, block_flat=True) # change
    demand1.update_total()

    grid1 = FutureGrid_v2(gd, base_year=2022)
    grid1.add_generators(year)
    grid1.drop_generators(year)
    grid1.year = year
    grid1.future = demand1

    grid1.run_storage_before_capacitydispatch(caps[year], max_rates[year], allow_negative=True)
    grid1.future.demand.demand = np.copy(grid1.storage_before.df.comb_demand_after_storage.values)
    grid1.run_dispatch(save_str, verbose=False, time_array=time_array, result_date=date, coal_downtime=False, year=year)
