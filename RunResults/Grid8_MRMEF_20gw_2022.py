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

baseyear = 2022
deltaMW_set = [20000]
year_set = np.arange(2023, 2038)

gd = pickle.load(open('../Data/generator_data_short_WECC_2022_cluster.obj', 'rb'))

date = '20240214'
folder = 'Results_MRMEF_outputs_2022BASE_'+date

if not os.path.isdir(folder):
    os.mkdir(folder)
    
signals = pd.read_csv('../Data/signals_df_mrmef_storagebefore_noevbase_2022BASE_20240212.csv', index_col=0)

time_array = np.arange(52)+1

# Storage:
scaling = pd.read_csv('../Data/scaling_generation_fractions_2024final.csv', index_col=0)
max_rates = {}
caps = {}
for i in scaling.index:
    max_rates[int(scaling.loc[i, 'Year'])] = scaling.loc[i, 'Battery MW']
    caps[int(scaling.loc[i, 'Year'])] = 4 * scaling.loc[i, 'Battery MW']


def run_dispatch_wrapped(save_str, gd, year, signal_here, time_array, date, hourly_signal_weekday=None, hourly_signal_weekend=None, baseyear=2022):
        
    demand1 = FutureDemand_v2(gd, year=year, base_year=baseyear)
    demand1.set_up_ready(evs=False, verbose=False, block=True, hourly_signal=signal_here, hourly_signal_weekday=hourly_signal_weekday, hourly_signal_weekend=hourly_signal_weekend) 
    demand1.update_total()

    grid1 = FutureGrid_v2(gd, base_year=baseyear)
    grid1.add_generators(year)
    grid1.drop_generators(year)
    grid1.year = year
    grid1.future = demand1

    grid1.run_storage_before_capacitydispatch(caps[year], max_rates[year], allow_negative=True)
    grid1.future.demand.demand = np.copy(grid1.storage_before.df.comb_demand_after_storage.values)
    grid1.run_dispatch(save_str, verbose=False, time_array=time_array, result_date=date, coal_downtime=False, year=year)

    dpdf = grid1.dp.df.copy(deep=True)
    
    return dpdf


year_signal_mapping = {2023:2023, 2024:2023, 2025:2023, 2026:2023, 2027:2023,
                       2028:2028, 2029:2028, 2030:2028, 2031:2028, 2032:2028, 
                       2033:2033, 2034:2033, 2035:2033, 2036:2033, 2037:2033}

for deltaMW in deltaMW_set:
    yearsahead = 5
    control = 'MRMEF_delta'+str(deltaMW)+'_'+str(yearsahead)+'yearsahead_weekdayweekend'

    dpdfs_all = {}
    for year in year_set:
        save_str = folder+'/controlled_block_'+control+'_year'+str(year)
        print(save_str)
        signalyear = year_signal_mapping[year]
        signal_here_weekday = signals['MRMEF_delta'+str(deltaMW)+'_startyear'+str(signalyear)+'_'+str(yearsahead)+'yearsahead'+'_weekday']
        signal_here_weekend = signals['MRMEF_delta'+str(deltaMW)+'_startyear'+str(signalyear)+'_'+str(yearsahead)+'yearsahead'+'_weekend']
        signal_here = None
        dpdfs_all[year] = run_dispatch_wrapped(save_str, gd, year, signal_here, time_array, date, hourly_signal_weekday=signal_here_weekday, hourly_signal_weekend=signal_here_weekend, baseyear=baseyear)


