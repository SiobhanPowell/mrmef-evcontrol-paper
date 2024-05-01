#!/usr/bin/env python
# coding: utf-8

# Based on code by Thomas Deetjen (https://github.com/tdeetjen/simple_dispatch), changed by Siobhan Powell

# In[1]:


# import os
# os.chdir('..')
import pickle
import scipy
import os.path
import pandas
from simple_dispatch import generatorData
from simple_dispatch import generatorDataShort
import time
import numpy as np


# # 2019

# In[2]:


run_year = 2019
ferc714_part2_schedule6_csv = '../Data/GridInputData/2019Final/Part 2 Schedule 6 - Balancing Authority Hourly System Lambda.csv'
ferc714IDs_csv='../Data/GridInputData/2019Final/Respondent IDs Cleaned.csv'
cems_folder_path ='../Data/GridInputData/2019Final/CEMS'
easiur_csv_path ='GridInputData/egrid_2019_plant_easiur.csv'
fuel_commodity_prices_xlsx = '../Data/GridInputData/2019Final/fuel_default_prices.xlsx'
egrid_data_xlsx = '../Data/GridInputData/2019Final/eGRID2019_data.xlsx'
eia923_schedule5_xlsx = '../Data/GridInputData/2019Final/EIA923_Schedules_2_3_4_5_M_12_2019_Final_Revision.xlsx'

nerc_region = 'WECC'

tic = time.time()
gd = generatorData(nerc_region, 
                   egrid_fname=egrid_data_xlsx, 
                   eia923_fname=eia923_schedule5_xlsx, 
                   ferc714IDs_fname=ferc714IDs_csv, 
                   ferc714_fname=ferc714_part2_schedule6_csv, 
                   cems_folder=cems_folder_path, 
                   easiur_fname=easiur_csv_path, 
                   include_easiur_damages=False, 
                   year=run_year, 
                   fuel_commodity_prices_excel_dir=fuel_commodity_prices_xlsx, 
                   hist_downtime=False, 
                   coal_min_downtime=12, 
                   cems_validation_run=False, 
                   tz_aware=True)   
toc = time.time()
print('Finished in '+str(np.round((toc-tic)/60, 2))+' minutes. Saving.')
gd_short = generatorDataShort(gd)
pickle.dump(gd_short, open('../Data/generator_data_short_WECC_2019_cluster.obj', 'wb'))
print('Saved.')


# In[3]:


# %pip install --upgrade pandas


# In[4]:
run_year = 2022
ferc714_part2_schedule6_csv = '../Data/GridInputData/2022Final/Part 2 Schedule 6 - Balancing Authority Hourly System Lambda.csv'
ferc714IDs_csv='../Data/GridInputData/2022Final/2022 Respondent IDs status.csv'
cems_folder_path ='../Data/GridInputData/2022Final/CEMS'
easiur_csv_path ='GridInputData/egrid_2022_plant_easiur.csv'
fuel_commodity_prices_xlsx = '../Data/GridInputData/2022Final/fuel_default_prices.xlsx'
egrid_data_xlsx = '../Data/GridInputData/2022Final/eGRID2022_data.xlsx'#egrid2021_data_untfixes.xlsx'
eia923_schedule5_xlsx = '../Data/GridInputData/2022Final/EIA923_Schedules_2_3_4_5_M_12_2022_Final.xlsx'

nerc_region = 'WECC'

tic = time.time()
gd = generatorData(nerc_region, 
                   egrid_fname=egrid_data_xlsx, 
                   eia923_fname=eia923_schedule5_xlsx, 
                   ferc714IDs_fname=ferc714IDs_csv, 
                   ferc714_fname=ferc714_part2_schedule6_csv, 
                   cems_folder=cems_folder_path, 
                   easiur_fname=easiur_csv_path, 
                   include_easiur_damages=False, 
                   year=run_year, 
                   fuel_commodity_prices_excel_dir=fuel_commodity_prices_xlsx, 
                   hist_downtime=False, 
                   coal_min_downtime=12, 
                   cems_validation_run=False, 
                   tz_aware=True)   
toc = time.time()


# In[5]:

print('Finished in '+str(np.round((toc-tic)/60, 2))+' minutes. Saving.')

# In[6]:


gd_short = generatorDataShort(gd)


# In[8]:


pickle.dump(gd_short, open('../Data/generator_data_short_WECC_2022_cluster.obj', 'wb'))
print('Saved.')

