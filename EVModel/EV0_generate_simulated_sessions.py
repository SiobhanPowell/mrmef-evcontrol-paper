import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from speech_classes import SPEECh
from speech_classes import SPEEChGeneralConfiguration
from speech_classes import LoadProfile
from speech_classes import Plotting
from speech_classes import DataSetConfigurations

scenario_name = 'HighHome'
remove_timers = True
utility_region = 'PGE'
tz_aware = True

#################
weekday='weekday'

sessions_df_home = pd.DataFrame(columns=['start_seconds', 'Session Time (secs)', 'Energy (kWh)'])
sessions_df_work = pd.DataFrame(columns=['start_seconds', 'Session Time (secs)', 'Energy (kWh)'])

state_list = ['CA', 'OR', 'WA', 'ID', 'MT', 'WY', 'NV', 'UT', 'CO', 'NM', 'AZ']
for state in state_list:
    data = DataSetConfigurations(data_set='CP')
    speech = SPEECh(data=data, penetration_level=1.0, outside_california=True, states=[state])
    speech.pa_ih(scenario=scenario_name) # define the access scenario
    speech.pg_multiple_regions(region_type='State', region_value_list=[state])
    config = SPEEChGeneralConfiguration(speech, remove_timers=remove_timers, utility_region=utility_region)
    # config.run_all(verbose=False, weekday='weekday')
    for g in range(config.speech.data.ng):
        model = LoadProfile(config, config.group_configs[g], weekday=weekday)
        model.calculate_load()
        if 'home_l2' in model.gmm_outputs.keys():
            sessions_df_home = pd.concat((sessions_df_home, pd.DataFrame({'start_seconds': model.gmm_outputs['home_l2'][:, 0], 
                                                                          'Energy (kWh)': model.gmm_outputs['home_l2'][:, 1],
                                                                          'Session Time (secs)': model.gmm_outputs['home_l2'][:, 2]})), axis=0, ignore_index=True)
        if 'work_l2' in model.gmm_outputs.keys():
            sessions_df_work = pd.concat((sessions_df_work, pd.DataFrame({'start_seconds': model.gmm_outputs['work_l2'][:, 0], 
                                                                          'Energy (kWh)': model.gmm_outputs['work_l2'][:, 1],
                                                                          'Session Time (secs)': model.gmm_outputs['work_l2'][:, 2]})), axis=0, ignore_index=True)
            
    sessions_df_home.to_csv('Data/simulated_home_sessions_weekday_20230303.csv')
    sessions_df_work.to_csv('Data/simulated_work_sessions_weekday_20230303.csv')
        

#################        
weekday='weekend'

sessions_df_home = pd.DataFrame(columns=['start_seconds', 'Session Time (secs)', 'Energy (kWh)'])
sessions_df_work = pd.DataFrame(columns=['start_seconds', 'Session Time (secs)', 'Energy (kWh)'])

state_list = ['CA', 'OR', 'WA', 'ID', 'MT', 'WY', 'NV', 'UT', 'CO', 'NM', 'AZ']
for state in state_list:
    data = DataSetConfigurations(data_set='CP')
    speech = SPEECh(data=data, penetration_level=1.0, outside_california=True, states=[state])
    speech.pa_ih(scenario=scenario_name) # define the access scenario
    speech.pg_multiple_regions(region_type='State', region_value_list=[state])
    config = SPEEChGeneralConfiguration(speech, remove_timers=remove_timers, utility_region=utility_region)
    # config.run_all(verbose=False, weekday='weekday')
    for g in range(config.speech.data.ng):
        model = LoadProfile(config, config.group_configs[g], weekday=weekday)
        model.calculate_load()
        if 'home_l2' in model.gmm_outputs.keys():
            sessions_df_home = pd.concat((sessions_df_home, pd.DataFrame({'start_seconds': model.gmm_outputs['home_l2'][:, 0], 
                                                                          'Energy (kWh)': model.gmm_outputs['home_l2'][:, 1],
                                                                          'Session Time (secs)': model.gmm_outputs['home_l2'][:, 2]})), axis=0, ignore_index=True)
        if 'work_l2' in model.gmm_outputs.keys():
            sessions_df_work = pd.concat((sessions_df_work, pd.DataFrame({'start_seconds': model.gmm_outputs['work_l2'][:, 0], 
                                                                          'Energy (kWh)': model.gmm_outputs['work_l2'][:, 1],
                                                                          'Session Time (secs)': model.gmm_outputs['work_l2'][:, 2]})), axis=0, ignore_index=True)
            
    sessions_df_home.to_csv('Data/simulated_home_sessions_weekend_20230303.csv')
    sessions_df_work.to_csv('Data/simulated_work_sessions_weekend_20230303.csv')
        