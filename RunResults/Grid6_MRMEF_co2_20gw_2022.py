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

year_set = np.arange(2023, 2038)

deltaMW_set = [20000]
co2_price_dol_per_kg = 0.1 

gd = pickle.load(open('../Data/generator_data_short_WECC_2022_cluster.obj', 'rb'))

date = '20240214'
folder2 = 'Results_MRMEF_CO2_2022BASE_'+date
if not os.path.isdir(folder2):
    os.mkdir(folder2)
    
scaling = pd.read_csv('../Data/scaling_generation_fractions_2024final.csv', index_col=0)
max_rates = {}
caps = {}
for i in scaling.index:
    max_rates[int(scaling.loc[i, 'Year'])] = scaling.loc[i, 'Battery MW']
    caps[int(scaling.loc[i, 'Year'])] = 4 * scaling.loc[i, 'Battery MW']

time_array = np.arange(52)+1

# MRMEF
for deltaMW in deltaMW_set:
    for year in year_set:
        for weekday in ['weekday', 'weekend']:
            for hour in np.arange(0, 24):
                save_str = folder2+'/MRMEF_deltaMW'+str(deltaMW)+'_hour'+str(hour)+'_year'+str(year)+'_'+weekday+'_co2price_01dolperkg'+'_storagebefore'
                print(save_str)

                demand1 = FutureDemand_v2(gd, year=year, base_year=2022)
                demand1.set_up_ready(evs=False)
                demand1.delta_demand_hour(hour, weekday=weekday, deltaMW=deltaMW)
                demand1.update_total()

                grid1 = FutureGrid_v2(gd, base_year=2022)
                grid1.add_generators(year)
                grid1.drop_generators(year)
                grid1.year = year
                grid1.future = demand1
                
                grid1.run_storage_before_capacitydispatch(caps[year], max_rates[year], allow_negative=True)
                grid1.future.demand.demand = np.copy(grid1.storage_before.df.comb_demand_after_storage.values)
                grid1.run_dispatch(save_str, verbose=False, time_array=time_array, result_date=date, coal_downtime=False, year=year, co2_dol_per_kg=co2_price_dol_per_kg)
