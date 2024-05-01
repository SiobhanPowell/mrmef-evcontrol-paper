"""
This code is an extension of that posted on https://github.com/SiobhanPowell/speech-grid-impact. It was originally developed in 2021 and updated in 2023 by Siobhan Powell and Sonia Martin. 
""" 


import pandas as pd
import numpy as np
import copy
from simple_dispatch import StorageModel
from simple_dispatch import generatorData
from simple_dispatch import bidStack
from simple_dispatch import dispatch
from simple_dispatch import generatorDataShort
import scipy
import cvxpy as cvx
import datetime

        
class FutureDemand_v2(object):
    """By Siobhan Powell. This class manages the model of the future grid demand.

    :param baseline_demand: Demand data before making any adjustments
    :type baseline_demand: Dataframe
    :param demand: Modeled future demand
    :type demand: Dataframe

    :param year: Year for the future grid
    :type year: int

    :param all_generation: Generation from non-fossil fuel sources in 2019, collected from the EIA.
    :type all_generation: Dataframe
    :param not_combustion: Generation from non-fossil fuel sources in 2019
    :type not_combustion: Dataframe

    :param electrification_scaling: Amount to scale each year for electrification in other sectors
    :type electrification_scaling: dict
    :param solar_multiplier: Amount to scale solar
    :type solar_multiplier: dict
    :param wind_multiplier: Amount to scale wind
    :type wind_multiplier: dict

    :param ev_load: Weekday EV demand, unscaled
    :type ev_load: Dataframe
    :param ev_load_add: Weekday EV demand, scaled by pen level
    :type ev_load_add: Dataframe

    :param ev_load_weekend: Weekend EV demand, unscaled
    :type ev_load_weekend: Dataframe
    :param ev_load_weekend_add: Weekend EV demand, scaled by pen level
    :type ev_load_weekend_add: Dataframe
    """

    def __init__(self, gd_short, year=2030, base_year=2019):

        self.year = year
        self.base_year = base_year
        self.baseline_demand = gd_short.demand_data.copy(deep=True)
        
        self.all_generation = pd.read_csv('../Data/region_eia_generation_data_'+str(base_year)+'.csv', index_col=0)
        
        self.baseline_demand['datetime'] = pd.to_datetime(self.baseline_demand['datetime'])
        self.baseline_demand['total_incl_noncombustion'] = self.baseline_demand['demand'].values + self.all_generation['WECC_notcombustion'].values
        self.not_combustion = pd.DataFrame({'dt': self.all_generation['dt'], 'generation': self.all_generation['WECC_notcombustion']})

        self.demand = self.baseline_demand.copy(deep=True)

        self.study_year_min = 2023
        self.study_year_max = 2038
        self.study_year_range = np.arange(self.study_year_min, self.study_year_max)
        
        self.scaling_df = pd.read_csv('../Data/scaling_generation_fractions_2024final.csv', index_col=0)
     
        self.ev_multipliers()
        self.electrification_multipliers()
        self.renewables_multipliers()
        
        self.ev_load_weekend = None
        self.ev_load_weekend_add = None
        self.day_week_mapping = None
        
    def ev_multipliers(self):
        self.ev_add_scaling = {2019:0.01, 2035:0.5}
        m = (self.ev_add_scaling[2035] - self.ev_add_scaling[2019]) / (2035-2019)
        for year in np.arange(2020, self.study_year_max):
            self.ev_add_scaling[int(year)] = self.ev_add_scaling[2019] + m * (year - 2019)
    
    def electrification_multipliers(self):
        
        self.electrification_scaling = {}
        col = 'Demand Over '+str(self.base_year)
        for i in self.scaling_df.index:
            self.electrification_scaling[int(self.scaling_df.loc[i, 'Year'])] = self.scaling_df.loc[i, col]
            
    def renewables_multipliers(self):
        
        self.solar_multiplier = {}
        col1 = 'Solar Over '+str(self.base_year)
        self.wind_multiplier = {}
        col2 = 'Wind Over '+str(self.base_year)
        self.hydro_multiplier = {}
        col3 = 'Hydro Over '+str(self.base_year)
        for i in self.scaling_df.index:
            self.solar_multiplier[int(self.scaling_df.loc[i, 'Year'])] = self.scaling_df.loc[i, col1]
            self.wind_multiplier[int(self.scaling_df.loc[i, 'Year'])] = self.scaling_df.loc[i, col2]
            self.hydro_multiplier[int(self.scaling_df.loc[i, 'Year'])] = self.scaling_df.loc[i, col3]
            
    def set_up_ready(self, evs=True, ev_name='HighHome', ev_uncontrolled=False, ev_control_name_weekday='AEF_weekday_2020', ev_control_name_weekend='AEF_weekend_2020', ev_reg='nonreg', ev_norm=False, verbose=False, block=False, hourly_signal=None, block_flat=False, block_weekly=False, block_weekly_signals=None, weekly_signals_key='AEF_2020', block_worst=False, hourly_signal_weekday=None, hourly_signal_weekend=None, ev_control_date='20230731', co2price=False, ev15min=False):
        
        self.electrification()
        self.solar()
        self.wind()
        self.hydro()
        self.geothermal()
        self.nuclear()
        
        if evs:
            self.evs(ev_name=ev_name, uncontrolled=ev_uncontrolled, control_name_weekday=ev_control_name_weekday, 
                     control_name_weekend=ev_control_name_weekend, reg=ev_reg, norm=ev_norm, ev_control_date=ev_control_date, co2price=co2price, ev15min=ev15min)
        if block:
            self.block_equivalent(hourly_signal=hourly_signal, flat=block_flat, weekly=block_weekly, weekly_signals=block_weekly_signals, weekly_signals_key=weekly_signals_key, worst=block_worst, hourly_signal_weekday=hourly_signal_weekday, hourly_signal_weekend=hourly_signal_weekend)
        if verbose:
            print('After EVs before update:', self.demand.total_incl_noncombustion.sum(), self.demand.demand.sum())
        self.update_total()
        if verbose:
            print('After update:', self.demand.total_incl_noncombustion.sum(), self.demand.demand.sum())

    def electrification(self):

        self.demand['demand'] = self.electrification_scaling[self.year] * self.baseline_demand['total_incl_noncombustion'] - self.all_generation['WECC_notcombustion'].values

    def solar(self):

        solar_2019 = self.all_generation['WECC_SUN'].values
        self.not_combustion['generation'] += (self.solar_multiplier[self.year]-1) * solar_2019
        self.demand['demand'] -= (self.solar_multiplier[self.year]-1) * solar_2019

    def wind(self):

        wind_2019 = self.all_generation['WECC_WND'].values
        self.not_combustion['generation'] += (self.wind_multiplier[self.year]-1) * wind_2019
        self.demand['demand'] -= (self.wind_multiplier[self.year]-1) * wind_2019
        
    def hydro(self):
        
        hydro_2019 = self.all_generation['WECC_WAT'].values
        self.not_combustion['generation'] += (self.hydro_multiplier[self.year]-1) * hydro_2019
        self.demand['demand'] -= (self.hydro_multiplier[self.year]-1) * hydro_2019
        
    def geothermal(self):
        
        geothermal_2019 = self.all_generation['WECC_OTH_GEO'].values # base year: 2021
        # starting 2021 namepcap = 3838
        # Additions: 29 MW in 2022, 439.5 in 2023, 44 in 2026, 801 in 2031
        
        if self.base_year == 2019:
            base_val = 3974
        elif self.base_year == 2022:
            base_val = 4018
        
        added_val = 0
        if self.year in [2020, 2021]:
            added_val = 15
        elif self.year in [2022]:
            added_val = 15 + 29
        elif self.year in [2023, 2024, 2025]:
            added_val = 15 + 29 + 439.5
        elif self.year in [2026, 2027, 2028, 2029, 2030]:
            added_val = 15 + 29 + 439.5 + 44
        elif self.year >= 2031:
            added_val = 15 + 29 + 439.5 + 44 + 801
        
        added_geothermal =  (added_val/base_val) * geothermal_2019 

        self.not_combustion['generation'] += added_geothermal
        self.demand['demand'] -= added_geothermal
        
    def nuclear(self):
        
        # Diablo canyon retiring 2 1150 MW units in 2030
        # 83% capacity factor in 2019 https://www.pgecorp.com/corp_responsibility/reports/2021/pf07_nuclear_operations.html#:~:text=Diablo%20Canyon%20continues%20to%20demonstrate,factor%20of%2083%25%20during%202020.
        if self.year > 2030:
            self.not_combustion['generation'] -= 0.83 * 2 * 1150 * np.ones((8760,))
            self.demand['demand'] += 0.83 * 2 * 1150 * np.ones((8760,))
        
    def integrate_hourly(self, minute_profile, ev15min=False):
        
        if ev15min:
            hour_profile = np.zeros((24,))
            for h in range(24):
                hour_profile[h] = (1/4)*np.sum(minute_profile[np.arange(h*4, (h+1)*4)])
            
        else:
            hour_profile = np.zeros((24,))
            for h in range(24):
                hour_profile[h] = (1/60)*np.sum(minute_profile[np.arange(h*60, (h+1)*60)])
        
        return hour_profile
    
    def block_equivalent(self, hourly_signal, flat=False, weekly=False, weekly_signals=None, weekly_signals_key='AEF_2020', worst=False, hourly_signal_weekday=None, hourly_signal_weekend=None):
        
        pen_level = self.ev_add_scaling[int(self.year)]
        self.evs_update_status = True
        self.ev_pen_level = pen_level
        
        ref = pd.read_csv('../Data/BusinessAsUsual_100p_WECC_20211119.csv', index_col=0)
        ref['Total'] = ref.sum(axis=1)
        mwh_total = (1/1000)*(1/60)*ref['Total'].sum()
        
        if flat:
            cont_weekday = pen_level * mwh_total * (1/24) * np.ones((24, ))
        elif worst:
            worst_hour_order = [18, 19, 17, 20, 16, 21, 15, 22, 6, 14, 5, 23, 4, 13, 7, 0, 12, 3, 1, 11, 8, 2, 10, 9]
            cont_weekday = np.zeros((24, ))
            max_hours = int(pen_level * mwh_total / 20000) # number of hours at 20 GW
            remaining = pen_level * mwh_total - 20000*max_hours
            if max_hours > 0:
                for i in range(max_hours):
                    cont_weekday[worst_hour_order[i]] = 20000
                cont_weekday[worst_hour_order[i+1]] = remaining
            else:
                cont_weekday[worst_hour_order[0]] = remaining
        elif weekly:
            cont_weekday_dict = {}
            for week in np.arange(1, 53):
                hourly_signal = weekly_signals[weekly_signals_key+'_week'+str(week)].values
                print('Week: ', week, '. Optimizing weekday.')
                cont_demand = cvx.Variable(24)
                constraints = [cont_demand >= 0, cont_demand <= 20000, cvx.sum(cont_demand)==pen_level * mwh_total]
                objective = cvx.Minimize(cvx.sum(cvx.multiply(cont_demand, hourly_signal)))
                prob = cvx.Problem(objective, constraints)
                prob.solve()
                cont_weekday_dict[week] = np.copy(cont_demand.value)
        elif hourly_signal_weekday is not None:
            print('Optimizing weekday')
            cont_demand = cvx.Variable(24)
            constraints = [cont_demand >= 0, cont_demand <= 20000, cvx.sum(cont_demand)==pen_level * mwh_total]
            objective = cvx.Minimize(cvx.sum(cvx.multiply(cont_demand, hourly_signal_weekday)))
            prob = cvx.Problem(objective, constraints)
            prob.solve()

            cont_weekday = cont_demand.value
        else:
            print('Optimizing weekday')
            cont_demand = cvx.Variable(24)
            constraints = [cont_demand >= 0, cont_demand <= 20000, cvx.sum(cont_demand)==pen_level * mwh_total]
            objective = cvx.Minimize(cvx.sum(cvx.multiply(cont_demand, hourly_signal)))
            prob = cvx.Problem(objective, constraints)
            prob.solve()

            cont_weekday = cont_demand.value
        
        ref_we = pd.read_csv('../Data/BusinessAsUsual_100p_weekend_WECC_20211119.csv', index_col=0)                       
        ref_we['Total'] = ref_we.sum(axis=1)
        mwh_total_we = (1/1000)*(1/60)*ref_we['Total'].sum()

        if flat:
            cont_weekend = pen_level * mwh_total_we * (1/24) * np.ones((24, ))
        elif weekly:
            cont_weekend_dict = {}
            for week in np.arange(1, 53):
                hourly_signal = weekly_signals[weekly_signals_key+'_week'+str(week)].values
                print('Week: ', week, '. Optimizing weekend.')
                cont_demand_we = cvx.Variable(24)
                constraints = [cont_demand_we >= 0, cont_demand_we <= 20000, cvx.sum(cont_demand_we)==pen_level * mwh_total_we]
                objective = cvx.Minimize(cvx.sum(cvx.multiply(cont_demand_we, hourly_signal)))
                prob = cvx.Problem(objective, constraints)
                prob.solve()
                cont_weekend_dict[week] = np.copy(cont_demand_we.value)
        elif worst:
            worst_hour_order = [18, 19, 17, 20, 16, 21, 15, 22, 6, 14, 5, 23, 4, 13, 7, 0, 12, 3, 1, 11, 8, 2, 10, 9]
            cont_weekend= np.zeros((24, ))
            max_hours = int(pen_level * mwh_total_we / 20000) # number of hours at 20 GW
            remaining = pen_level * mwh_total_we - 20000*max_hours
            if max_hours > 0:
                for i in range(max_hours):
                    cont_weekend[worst_hour_order[i]] = 20000
                cont_weekend[worst_hour_order[i+1]] = remaining
            else:
                cont_weekend[worst_hour_order[0]] = remaining
        elif hourly_signal_weekend is not None:
            print('Optimizing weekend')
            cont_demand_we = cvx.Variable(24)
            constraints = [cont_demand_we >= 0, cont_demand_we <= 20000, cvx.sum(cont_demand_we)==pen_level * mwh_total_we]
            objective = cvx.Minimize(cvx.sum(cvx.multiply(cont_demand_we, hourly_signal_weekend)))
            prob = cvx.Problem(objective, constraints)
            prob.solve()

            cont_weekend = cont_demand_we.value
        else:
            print('Optimizing weekend')
            cont_demand_we = cvx.Variable(24)
            constraints = [cont_demand_we >= 0, cont_demand_we <= 20000, cvx.sum(cont_demand_we)==pen_level * mwh_total_we]
            objective = cvx.Minimize(cvx.sum(cvx.multiply(cont_demand_we, hourly_signal)))
            prob = cvx.Problem(objective, constraints)
            prob.solve()

            cont_weekend = cont_demand_we.value
        
        if weekly:
            self.controlled_blocks = {'weekday':cont_weekday_dict, 'weekend':cont_weekend_dict}
        else:
            self.controlled_blocks = {'weekday':cont_weekday, 'weekend':cont_weekend}
        
        if weekly:
            if self.day_week_mapping is None:
                self.day_week_mapping = pd.read_csv('../Data/day_week_number_mapping.csv', index_col=0)
            for i in range(365):
                week_here = self.day_week_mapping[self.day_week_mapping['Day']==i+1]['WeekNumber'].values[0]
                if pd.to_datetime(self.demand.loc[24*i, 'datetime']).weekday() in [0, 1, 2, 3, 4]:
                    self.demand.loc[24*i+np.arange(0, 24), 'demand'] += np.copy(cont_weekday_dict[week_here])
                else:
                    self.demand.loc[24*i+np.arange(0, 24), 'demand'] += np.copy(cont_weekend_dict[week_here])
        else:
            self.ev_load_add = np.copy(cont_weekday)
            self.ev_load_weekend_add = np.copy(cont_weekend)
            for i in range(365):
                if pd.to_datetime(self.demand.loc[24*i, 'datetime']).weekday() in [0, 1, 2, 3, 4]:
                    self.demand.loc[24*i+np.arange(0, 24), 'demand'] += self.ev_load_add
                else:
                    self.demand.loc[24*i+np.arange(0, 24), 'demand'] += self.ev_load_weekend_add
        
    def evs(self, ev_name='HighHome', uncontrolled = False, control_name_weekday='AEF_weekday_2020', control_name_weekend='AEF_weekend_2020', reg='nonreg', norm=False, ev_control_date='20230731', co2price=False, ev15min=False):
        
        pen_level = self.ev_add_scaling[int(self.year)]

        self.evs_update_status = True
        self.ev_pen_level = pen_level
        
        if ev15min:
            ref = pd.read_csv('../Data/EVProfiles/uncontrolled_15min_UniversalHome_100p_weekday_'+ev_control_date+'.csv', index_col=0)
            ref_we = pd.read_csv('../Data/EVProfiles/uncontrolled_15min_UniversalHome_100p_weekend_'+ev_control_date+'.csv', index_col=0)
        else:
            ref = pd.read_csv('../Data/EVProfiles/uncontrolled_UniversalHome_100p_weekday_'+ev_control_date+'.csv', index_col=0)
            ref_we = pd.read_csv('../Data/EVProfiles/uncontrolled_UniversalHome_100p_weekend_'+ev_control_date+'.csv', index_col=0)

        if uncontrolled:
            if ev15min:
                basecase = pd.read_csv('../Data/EVProfiles/uncontrolled_15min_'+ev_name+'_100p_weekday_'+ev_control_date+'.csv', index_col=0)
                basecase_we = pd.read_csv('../Data/EVProfiles/uncontrolled_15min_'+ev_name+'_100p_weekend_'+ev_control_date+'.csv', index_col=0)
            else:
                basecase = pd.read_csv('Data/EVProfiles/uncontrolled_'+ev_name+'_100p_weekday_'+ev_control_date+'.csv', index_col=0)
                basecase_we = pd.read_csv('Data/EVProfiles/uncontrolled_'+ev_name+'_100p_weekend_'+ev_control_date+'.csv', index_col=0)

        else:
            if co2price:
                if ev15min:
                    basecase = pd.read_csv('../Data/EVProfiles/controlled_15min_co2_'+control_name_weekday+'_'+ev_name+'_100p_'+reg+'_'+ev_control_date+'.csv', index_col=0)
                    basecase_we = pd.read_csv('../Data/EVProfiles/controlled_15min_co2_'+control_name_weekend+'_'+ev_name+'_100p_'+reg+'_'+ev_control_date+'.csv', index_col=0)
                else:
                    basecase = pd.read_csv('../Data/EVProfiles/controlled_co2_'+control_name_weekday+'_'+ev_name+'_100p_'+reg+'_'+ev_control_date+'.csv', index_col=0)
                    basecase_we = pd.read_csv('../Data/EVProfiles/controlled_co2_'+control_name_weekend+'_'+ev_name+'_100p_'+reg+'_'+ev_control_date+'.csv', index_col=0)
            else:
                if ev15min:
                    basecase = pd.read_csv('../Data/EVProfiles/controlled_15min_'+control_name_weekday+'_'+ev_name+'_100p_'+reg+'_'+ev_control_date+'.csv', index_col=0)
                    basecase_we = pd.read_csv('../Data/EVProfiles/controlled_15min_'+control_name_weekend+'_'+ev_name+'_100p_'+reg+'_'+ev_control_date+'.csv', index_col=0)
                else:
                    basecase = pd.read_csv('../Data/EVProfiles/controlled_'+control_name_weekday+'_'+ev_name+'_100p_'+reg+'_'+ev_control_date+'.csv', index_col=0)
                    basecase_we = pd.read_csv('../Data/EVProfiles/controlled_'+control_name_weekend+'_'+ev_name+'_100p_'+reg+'_'+ev_control_date+'.csv', index_col=0)
#         ref = pd.read_csv('Data/'+'BusinessAsUsual'+'_100p_WECC_20211119.csv', index_col=0)
#         basecase_we = pd.read_csv('Data/uncontrolled_'+control_name+'_100p_weekend_WECC_20211119.csv', index_col=0)                       
        basecase['Total'] = basecase.sum(axis=1)
        basecase_we['Total'] = basecase_we.sum(axis=1)
        ref['Total'] = ref.sum(axis=1)
        ref_we['Total'] = ref_we.sum(axis=1)
        
        bc_hourly = self.integrate_hourly(basecase['Total'].values, ev15min=ev15min)
        bc_we_hourly = self.integrate_hourly(basecase_we['Total'].values, ev15min=ev15min)
        ref_hourly = self.integrate_hourly(ref['Total'].values, ev15min=ev15min)
        ref_we_hourly = self.integrate_hourly(ref_we['Total'].values, ev15min=ev15min)
                
#         basecase['Total_Norm'] = basecase['Total'] * (ref.Total.sum() / basecase.Total.sum())
        bc_norm_hourly = bc_hourly * (np.sum(ref_hourly) / np.sum(bc_hourly))
        bc_we_norm_hourly = bc_we_hourly * (np.sum(ref_we_hourly) / np.sum(bc_we_hourly))
        
        # apply pen level and convert to MW
        if norm:
            self.ev_load_add = pen_level * (1/1000) * np.copy(bc_norm_hourly)#basecase['Total_Norm'].values)
            self.ev_load_weekend_add = pen_level * (1/1000) * np.copy(bc_we_norm_hourly)#basecase_we['Total'].values)
        else:
            self.ev_load_add = pen_level * (1/1000) * np.copy(bc_hourly)#basecase['Total'].values)
            self.ev_load_weekend_add = pen_level * (1/1000) * np.copy(bc_we_hourly)#basecase_we['Total'].values)

        for i in range(365):
            if pd.to_datetime(self.demand.loc[24*i, 'datetime']).weekday() in [0, 1, 2, 3, 4]:
                self.demand.loc[24*i+np.arange(0, 24), 'demand'] += self.ev_load_add
            else:
                self.demand.loc[24*i+np.arange(0, 24), 'demand'] += self.ev_load_weekend_add

    def delta_demand_hour(self, hour, weekday='weekday', deltaMW=1000):

        add_vec = np.zeros((24, ))
        add_vec[hour] = deltaMW
        
        for i in range(365):
            if pd.to_datetime(self.demand.loc[24*i, 'datetime']).weekday() in [0, 1, 2, 3, 4]:
                if weekday == 'weekday':
                    self.demand.loc[24*i+np.arange(0, 24), 'demand'] += add_vec
            else:
                if weekday == 'weekend':
                    self.demand.loc[24*i+np.arange(0, 24), 'demand'] += add_vec
                    
    def delta_demand_all_hours(self, weekday='weekday', deltaMW=1000):

        add_vec = deltaMW * np.ones((24, ))
        
        for i in range(365):
            if pd.to_datetime(self.demand.loc[24*i, 'datetime']).weekday() in [0, 1, 2, 3, 4]:
                if weekday == 'weekday':
                    self.demand.loc[24*i+np.arange(0, 24), 'demand'] += add_vec
            else:
                if weekday == 'weekend':
                    self.demand.loc[24*i+np.arange(0, 24), 'demand'] += add_vec

                
    def update_total(self):

        self.demand['total_incl_noncombustion'] = self.demand['demand'] + self.not_combustion['generation']


class FutureGrid_v2(object):
    """By Siobhan Powell. This class manages the model of the future grid and implements dispatch / capacity calculations.

    :param gd_short: The generator model
    :type gd_short: An object of class `generatorDataShort` from `simple_dispatch.py`

    :param retirements_df: Information about which generators are retired in each year
    :type retirements_df: Dataframe
    :param additions_df: Information about which generators are added each year
    :type additions_df: Dataframe

    :param year: Year for the future grid
    :type year: int

    :param future: Future grid demand, including EV demand
    :type future: An object of class `FutureDemand` from later in this file

    :param stor_df: Demand that needs to be met by storage_before; passed to storage model object
    :type stor_df: Dataframe
    :param storage_before: Storage model
    :type storage_before: An object of the class `StorageModel` from `simple_dispatch.py`
    
    :param storage_after: Storage model
    :type storage_after: An object of the class `StorageModel` from `simple_dispatch.py`

    :param bs: Bidstack
    :type bs: An object of the class `bidStack` by Thomas Deetjen from `simple_dispatch.py`
    :param dp: Dispatch
    :type dp: An object of the class `dispatch` by Thomas Deetjen from `simple_dispatch.py`
    """
    
    def __init__(self, gd_short, gen_limits=None, return_gen_limits=False, base_year=2019, study_year_min=2023, study_year_max=2038):
        
        self.gd_short = gd_short
        self.gd_short_original = copy.deepcopy(gd_short)
        
        self.base_year = base_year
        
        self.retirements_df = pd.read_csv('../Data/scheduled_retirements_2024final.csv', index_col=0)
        self.additions_df = pd.read_csv('../Data/generator_additions_2024final.csv', index_col=0)
        if 'retirement_year' in self.additions_df.columns:
            self.additions_df.loc[self.additions_df[self.additions_df['retirement_year'].isna()].index, 'retirement_year'] = 2100
        else:
            self.additions_df['retirement_year'] = 2100

        self.year = None
        self.future = None
        self.stor_df = None
        self.storage = None
        self.bs = None
        self.dp = None
        
        self.study_year_min = study_year_min#2023
        self.study_year_max = study_year_max#2037
        self.study_year_range = np.arange(self.study_year_min, self.study_year_max)
        
        if return_gen_limits:
            self.gen_limits = {}
            for year in self.study_year_range:
                self.gen_limits[year] = pd.DataFrame({'datetime':pd.date_range(datetime.datetime(self.base_year,1,1,0,0,0), datetime.datetime(self.base_year,12,31,23,0,0), freq='h'),'clipval':67445*np.ones((8760,))})
        else:
            if gen_limits is None:
                self.gen_limits = {}
                for year in self.study_year_range:
                    self.gen_limits[year] = pd.read_csv('../Data/gen_limits_'+str(year)+'_baseyear'+str(self.base_year)+'.csv', index_col=0)
#                 if base_year == 2022:
#                     for year in np.arange(2022, 2038):
#                         self.gen_limits[year] = pd.read_csv('../Data/gen_limits_'+str(year)+'_baseyear2022.csv', index_col=0)
#                 elif base_year == 2021:
#                     for year in np.arange(2021, 2037):
#                         self.gen_limits[year] = pd.read_csv('../Data/gen_limits_'+str(year)+'.csv', index_col=0)
#                 else:
#                     print('Unknown base year.')

    def add_generators(self, future_year):
        """Duplicate generators to simulate new additions in the future WECC grid."""
        
        gd_short_final = copy.deepcopy(self.gd_short)
#         added_units = self.additions_df.loc[(self.additions_df['Year']<future_year)]['orispl_unit'].values
        added_units = self.additions_df.loc[(self.additions_df['Year']<future_year)&(self.additions_df['Year']>=self.base_year)&(future_year<=self.additions_df['retirement_year'])]['orispl_unit'].values
        for i, val in enumerate(added_units):
            idx = len(gd_short_final.df)
            loc1 = gd_short_final.df[gd_short_final.df['orispl_unit']==val].index
            gd_short_final.df = pd.concat((gd_short_final.df, gd_short_final.df.loc[loc1]), ignore_index=True)
            gd_short_final.df.loc[idx, 'orispl_unit'] = 'added_'+str(i)
            
        if (future_year <= 2024) and (self.base_year <= 2020):
            # special case of two small coal plants (no existing coal plants small enough to just duplicate normally):
            # special new coal plant: 56224_001 / 2.4, built 2020, retires in 2024
            val = '56224_001'
            mult = 2.4
            idx = len(gd_short_final.df)
            loc1 = gd_short_final.df[gd_short_final.df['orispl_unit']==val].index
            tmp = gd_short_final.df.loc[loc1]
            tmp['mw'] = tmp['mw'] / mult
            for i in np.arange(1, 53):
                tmp['mw'+str(int(i))] / tmp['mw'+str(int(i))] / mult
            tmp['min_out'] = tmp['min_out'] / mult
            gd_short_final.df = pd.concat((gd_short_final.df, tmp), ignore_index=True)
            gd_short_final.df.loc[idx, 'orispl_unit'] = 'added_'+str(i+1)
            if future_year <= 2023:
                # special new coal plant: 56224_001 / 8, built in 2020, retires in 2023 (same reason)
                val = '56224_001'
                mult = 8.0
                idx = len(gd_short_final.df)
                loc1 = gd_short_final.df[gd_short_final.df['orispl_unit']==val].index
                tmp = gd_short_final.df.loc[loc1]
                tmp['mw'] = tmp['mw'] / mult
                for i in np.arange(1, 53):
                    tmp['mw'+str(int(i))] / tmp['mw'+str(int(i))] / mult
                tmp['min_out'] = tmp['min_out'] / mult
                gd_short_final.df = pd.concat((gd_short_final.df, tmp), ignore_index=True)
                gd_short_final.df.loc[idx, 'orispl_unit'] = 'added_'+str(i+2)
            
        self.gd_short = copy.deepcopy(gd_short_final)
        
    def drop_generators(self, future_year):
        """Drop generators to match announced retirements in the WECC grid."""
        
        gd_short_final = copy.deepcopy(self.gd_short)
        dropped_units = self.retirements_df[self.retirements_df['retirement_year']<future_year]['orispl_unit'].values
        gd_short_final.df = gd_short_final.df[~gd_short_final.df['orispl_unit'].isin(dropped_units)].copy(deep=True).reset_index(drop=True)
        
        self.gd_short = copy.deepcopy(gd_short_final)
        
#     def change_fuel_prices(self, future_year):
        
#         self.gd_short_originalfuelprices = copy.deepcopy(self.gd_short)
        
#         self.forecast = pd.read_excel('../Data/Table_3._Energy_Prices_by_Sector_and_Source.xlsx', sheet_name='PandasReady')
#         inds = self.gd_short.df.loc[self.gd_short.df.fuel_type=='gas'].index
#         for i in np.arange(1, 53):
#             self.gd_short.df.loc[inds, 'fuel_price'+str(int(i))] = self.forecast.loc[2, future_year]
#         inds = self.gd_short.df.loc[self.gd_short.df.fuel_type=='coal'].index
#         for i in np.arange(1, 53):
#             self.gd_short.df.loc[inds, 'fuel_price'+str(int(i))] = self.forecast.loc[3, future_year]
        
    def check_overgeneration(self, save_str=None, result_date=''):
        """Check for negative demand. Clip and save overgeneration amount."""

        if self.future.demand['demand'].min() < 0:
            if save_str is not None:
                self.future.demand.loc[self.future.demand['demand'] < 0].to_csv(save_str+'_overgeneration'+result_date+'.csv', index=None)
            self.future.demand['demand'] = self.future.demand['demand'].clip(0, 1e10)

    def run_storage_before_capacitydispatch(self, cap, max_rate, allow_negative=True):
        """If running storage on net demand before dispatch, do that here."""
    
        self.stor_df = pd.DataFrame({'datetime': pd.to_datetime(self.future.demand['datetime'].values),
                                     'total_demand': self.future.demand['demand'].values})
        self.storage_before = StorageModel(self.stor_df)
        self.storage_before.calculate_operation_beforecapacity(cap, max_rate, allow_negative=allow_negative)
        
    def pre_calculate_storage(self, save_str, result_date):
        
        self.future.demand['capacity_clipval'] = self.gen_limits[self.year].clipval
        self.future.demand['storage_demand'] = float(0)
        self.future.demand['unclipped_demand'] = np.copy(self.future.demand['demand'])

        inds = self.future.demand[self.future.demand.demand >= self.future.demand.capacity_clipval].index
        self.future.demand.loc[inds, 'storage_demand'] = self.future.demand.loc[inds, 'demand'].astype(float) - self.future.demand.loc[inds,'capacity_clipval'].astype(float)
        self.future.demand['demand'] = np.minimum(self.future.demand.demand, self.future.demand.capacity_clipval)
        
        storage_df = self.future.demand.loc[:, ['datetime', 'storage_demand', 'capacity_clipval', 'unclipped_demand']].rename(columns={'storage_demand':'demand'})
        self.storage_after = StorageModel(storage_df)
        self.storage_after.calculate_minbatt_forcapacity()
        print('Storage Rate Result:', int(self.storage_after.min_maxrate))
        print('Storage Capacity: ', int(self.storage_after.min_capacity))
        self.storage_after.df.to_csv(save_str+'_storage_after_'+result_date+'.csv', index=False)
        self.storage_stats = pd.DataFrame({'Storage Rate Result':int(self.storage_after.min_maxrate),'Storage Capacity':int(self.storage_after.min_capacity)}, index=[0])
        self.storage_stats.to_csv(save_str+'_storage_after_stats_'+result_date+'.csv', index=False)

        if self.storage_after.min_maxrate > 0:
            print('Scheduling extra storage.')
            self.storage_capacity_df = self.future.demand.loc[:, ['datetime', 'capacity_clipval', 'demand', 'storage_demand', 'unclipped_demand']].copy(deep=True)
            self.extra_storage = StorageModel(self.storage_capacity_df)
            self.extra_storage.schedule_extra_storage(cap=self.storage_after.min_capacity, max_rate=self.storage_after.min_maxrate)
            self.extra_storage.df.to_csv(save_str+'_extrastorage_scheduled_'+result_date+'.csv', index=False)
            
            self.future.demand.demand = np.copy(self.extra_storage.df.comb_demand_after_storage.values)
        
    def run_dispatch(self, save_str, result_date='20220223', co2_dol_per_kg=0, verbose=True, time_array=scipy.arange(52)+1, coal_downtime=True, force_storage=False, calculate_storage_size=True, return_gen_limits=False, year=None):
        """Run the dispatch. max_penlevel indicates whether storage will be needed or whether the model will break
        without it, but the try except clause will ensure the simulation is run if that is incorrect."""

        self.year = year
        self.bs = bidStack(self.gd_short, co2_dol_per_kg=co2_dol_per_kg, time=1, dropNucHydroGeo=True, include_min_output=False, mdt_weight=0.5, include_easiur=False) 
#         if storage_before:
#             self.future.demand.demand = np.copy(self.storage_before.df.comb_demand_after_storage.values)
        self.check_overgeneration(save_str=save_str, result_date=result_date)
        self.pre_calculate_storage(save_str=save_str, result_date=result_date)
        self.check_overgeneration(save_str=save_str+'_v2', result_date=result_date)
#         if force_storage:
        self.dp = dispatch(self.bs, self.future.demand, time_array=time_array, include_storage=True, return_generator_limits=return_gen_limits)
        self.dp.calcDispatchAll(verbose=verbose, coal_downtime=coal_downtime)
        self.dp.df.to_csv(save_str+'_withstorage'+'_dpdf_'+result_date+'.csv', index=False)
#         if calculate_storage_size:
#             self.storage_after = StorageModel(self.dp.storage_df)
#             self.storage_after.calculate_minbatt_forcapacity()
#             print('Storage Rate Result:', int(self.storage_after.min_maxrate))
#             print('Storage Capacity: ', int(self.storage_after.min_capacity))
#             self.storage_after.df.to_csv(save_str+'_storage_after_'+result_date+'.csv', index=False)
#             self.storage_stats = pd.DataFrame({'Storage Rate Result':int(self.storage_after.min_maxrate),'Storage Capacity':int(self.storage_after.min_capacity)}, index=[0])
#             self.storage_stats.to_csv(save_str+'_storage_after_stats_'+result_date+'.csv', index=False)


        