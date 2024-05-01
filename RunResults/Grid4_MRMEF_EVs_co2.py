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


gd = pickle.load(open('../Data/generator_data_short_WECC_2019_cluster.obj', 'rb'))

date = '20240208'
co2_dol_per_kg = 0.1
ev_control_date = '20240208'
folder = 'Results_MRMEF_EV_co2_'+date
if not os.path.isdir(folder):
    os.mkdir(folder)
    
time_array = np.arange(52)+1
year_set = np.arange(2023, 2038)

# Storage:
scaling = pd.read_csv('../Data/scaling_generation_fractions_2024final.csv', index_col=0)
max_rates = {}
caps = {}
for i in scaling.index:
    max_rates[int(scaling.loc[i, 'Year'])] = scaling.loc[i, 'Battery MW']
    caps[int(scaling.loc[i, 'Year'])] = 4 * scaling.loc[i, 'Battery MW']

def run_dispatch_wrapped(save_str, gd, year, signal_here, time_array, date, hourly_signal_weekday=None, hourly_signal_weekend=None, ev_uncontrolled=False, ev_name='HighHome', ev_control_name_weekday='AEF_weekday_2020', ev_control_name_weekend='AEF_weekend_2020', ev_reg='nonreg', ev_norm=True, ev_control_date='20230731', baseyear=2019, co2_dol_per_kg=0.1):
    
    demand1 = FutureDemand_v2(gd, year=year, base_year=baseyear)
    demand1.set_up_ready(evs=True, verbose=False, block=False, ev_name=ev_name, ev_uncontrolled=ev_uncontrolled, ev_control_name_weekday=ev_control_name_weekday, ev_control_name_weekend=ev_control_name_weekend, ev_reg=ev_reg, ev_norm=ev_norm, ev_control_date=ev_control_date, ev15min=True)
    demand1.update_total()

    grid1 = FutureGrid_v2(gd, base_year=baseyear)
    grid1.add_generators(year)
    grid1.drop_generators(year)
    grid1.year = year
    grid1.future = demand1

    grid1.run_storage_before_capacitydispatch(caps[year], max_rates[year], allow_negative=True)
    grid1.future.demand.demand = np.copy(grid1.storage_before.df.comb_demand_after_storage.values)
    grid1.run_dispatch(save_str, verbose=False, time_array=time_array, result_date=date, coal_downtime=False, year=year, co2_dol_per_kg=co2_dol_per_kg)

    dpdf = grid1.dp.df.copy(deep=True)
    
    return dpdf

year_signal_mapping = {2023:2023, 2024:2023, 2025:2023, 2026:2023, 2027:2023,
                       2028:2028, 2029:2028, 2030:2028, 2031:2028, 2032:2028, 
                       2033:2033, 2034:2033, 2035:2033, 2036:2033, 2037:2033}

for ev_name in ['UniversalHome', 'HighHome', 'LowHome_HighWork']:
    dpdfs_all = {}
    for year in year_set:
        save_str = folder+'/uncontrolled_EVs_'+ev_name+'_year'+str(year)
        print(save_str)
        signal_here = None
        dpdfs_all[year] = run_dispatch_wrapped(save_str, gd, year, signal_here, time_array, date, ev_uncontrolled=True, ev_name=ev_name, ev_norm=True, co2_dol_per_kg=co2_dol_per_kg, ev_control_date=ev_control_date)


for reg in ['nonreg']:
    for deltaMW in [5000, 10000]:
        for ev_name in ['UniversalHome', 'HighHome', 'LowHome_HighWork']:
            dpdfs_all = {}
            for year in year_set:
                save_str = folder+'/controlled_15min_co2_EVs_MRMEF_delta'+str(deltaMW)+'_5yearsahead_weekdayweekend_noevbase'+'_'+ev_name+'_'+reg+'_year'+str(year)
                print(save_str)
                signalyear = year_signal_mapping[year]
                ev_control_name_weekday = 'MRMEF_delta'+str(deltaMW)+'_startyear'+str(signalyear)+'_5yearsahead_weekday'
                ev_control_name_weekend = 'MRMEF_delta'+str(deltaMW)+'_startyear'+str(signalyear)+'_5yearsahead_weekend'
                signal_here = None
                dpdfs_all[year] = run_dispatch_wrapped(save_str, gd, year, signal_here, time_array, date, ev_uncontrolled=False, ev_name=ev_name, ev_control_name_weekday=ev_control_name_weekday, ev_control_name_weekend=ev_control_name_weekend, ev_reg=reg, ev_norm=True, ev_control_date=ev_control_date, co2_dol_per_kg=co2_dol_per_kg)


