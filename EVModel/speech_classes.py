""" SPEECH Model Extended
By: Siobhan Powell, 2021. 

This code is identical to that presented in https://github.com/SiobhanPowell/speech-grid-impact. Copyright (c) 2021, SiobhanPowell. All rights reserved.

Siobhan Powell, Gustavo Vianna Cezar, Liang Min, Ines Azevedo & Ram Rajagopal. (2021). SiobhanPowell/speech-grid-impact: (v1.0.0). Zenodo. doi: 10.5281/zenodo.5789549.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import os
from scipy.interpolate import interp1d


class DataSetConfigurations(object):
    """Store the information for each data set prepared.

    :param data_set: Name of data set
    :type data_set: str
    :param ng: Number of groups
    :type ng: int, optional

    :param categories: Categories / charging segments modeled with this data set.
    :type categories: list of strings
    :param labels: Formal names for the categories
    :type labels: list of strings
    :param colours: Colours used when plotting the categories
    :type colours: dict
    :param num_categories: Number of categories
    :type num_categories: int
    :param rates: Uncontrolled charging rate for each category, in kW
    :type rates: list of floats
    :param zkey_weekday: Name used on weekdays in the file storing P(z)
    :type zkey_weekday: str
    :param zkey_weekend: Name used on weekends in the file storing P(z)
    :type zkey_weekend: str

    :param cluster_reorder_dendtoac: Mapping of group numbers from how they appear in the dendrogram to the clustering.
    :type cluster_reorder_dendtoac: dict
    :param cluster_reorder_actodend: Inverse mapping of cluster_reorder_dendtoac
    :type cluster_reorder_actodend: dict

    :param gmm_names: Name for each category as it appears in the GMM file string
    :type gmm_names: dict
    :param start_time_scaler: Multiplier for GMM generated start times to put into seconds
    :type start_time_scaler: int
    :param start_mod: Modulo for shifting generated start times
    :type start_mod: int

    :param timers_dict: How components should be reweighted to remove timers
    :type timers_dict: dict of dicts
    :param shift_timers_dict: Which components should be shifted and to where when adjust timers by region
    :type shift_timers_dict: dict of dicts
    :param timer_cat: Which category is contains these timers
    :type timer_cat: str

    :param weekend_exists: Indicates whether the chosen data set and model includes weekends.
    :type weekend_exists: bool
    :param separate_weekend: Indicates whether a separate model should be used for weekends.
    :type separate_weekend: bool
    :param groups_with_separate_weekends: Which of the driver groups use a separate model for weekends.
    :type groups_with_separate_weekends: list

    """

    def __init__(self, data_set, ng=None):

        self.categories = []
        self.labels = []
        self.colours = {}
        self.num_categories = 0
        self.rates = []
        self.zkey_weekday = ''
        self.zkey_weekend = ''

        self.cluster_reorder_dendtoac = {}
        self.cluster_reorder_actodend = {}

        self.gmm_names = {}
        self.gmm_subfolder = ''
        self.start_time_scaler = 1
        self.start_mod = 1

        self.timers_dict = {}
        self.shift_timers_dict = {}
        self.weekend_timers_dict = None
        self.weekend_shift_timers_dict = None
        self.timer_cat = ''

        self.weekend_exists = True
        self.separate_weekend = False
        self.groups_with_separate_weekends = []

        if data_set in ['CP', 'CPLargeN', 'CP136']:
            self.data_set = 'CP136'
            self.ng = 136
        else:
            self.ng = ng
            self.data_set = data_set

        self.folder = 'Data/'+self.data_set+'/'

        if self.data_set == 'CP136':
            self.CP136()
        else:
            raise Exception('Unknown data set.')

        self.set_up_dendrogram_cluster_mapping()
        
    def CP136(self):
        """Separate drivers into energy buckets and cluster within each set."""

        self.categories = ['home_l1', 'home_l2', 'mud_l2', 'work_l2', 'public_l2', 'public_l3']
        self.labels = ['Residential L1', 'Residential L2', 'MUD L2', 'Workplace L2', 'Public L2', 'Public DCFC']
        self.colours = {'Residential L1': '#bf812d', 'Residential L2': '#dfc27d', 'MUD L2': '#f6e8c3', 'Workplace L2': '#80cdc1', 'Public L2': '#01665e', 'Public DCFC': '#003c30'}
        self.num_categories = 6
        self.rates = [1.4, 6.6, 6.6, 6.6, 6.6, 150]
        self.gmm_names = {i: i for i in self.categories}
        self.start_time_scaler = 1/60
        self.zkey_weekday = ' - Fraction of weekdays with session'
        self.zkey_weekend = ' - Fraction of weekenddays with session'
        self.start_mod = 24*3600
        self.timers_dict = {1: {6: 0, 1: 0.14849, 5: 0.03703, 7: 0.03703},
 2: {9: 0, 0: 0.23627, 2: 0.10077, 5: 0.29567, 7: 0.0427},
 7: {3: 0, 6: 0, 2: 0.225, 7: 0.075},
 15: {4: 0, 3: 0, 0: 0.42135, 7: 0.14317},
 25: {1: 0, 4: 0.13225, 6: 0.37482, 9: 0.02765},
 28: {3: 0, 1: 0.12786, 9: 0.33176},
 38: {4: 0, 1: 0, 0: 0.14853, 3: 0.04673, 5: 0.1479, 7: 0.24463, 9: 0.15194},
 43: {4: 0, 0: 0, 1: 0.22541, 6: 0.35318, 8: 0.04891},
 50: {0: 0, 5: 0.07659, 6: 0.3205, 9: 0.16399},
 52: {0: 0, 2: 0.09133, 3: 0.18267, 7: 0.274},
 53: {1: 0, 6: 0, 4: 0.15537, 7: 0.16155, 9: 0.31757},
 56: {8: 0, 1: 0.49652, 4: 0.1297, 5: 0.05828},
 63: {2: 0, 0: 0.10313, 6: 0.35076, 7: 0.21534, 9: 0.01554},
 69: {4: 0, 0: 0.03238, 3: 0.2275, 8: 0.25976},
 73: {4: 0, 6: 0, 0: 0.14756, 7: 0.34382, 9: 0.2165},
 75: {3: 0, 0: 0.34828, 6: 0.18934, 7: 0.11725, 8: 0.02806},
 79: {1: 0, 4: 0.22312, 7: 0.29517, 8: 0.07437},
 80: {1: 0, 4: 0, 6: 0, 7: 0, 8: 0, 9: 0, 2: 0.16234},
 90: {1: 0, 2: 0, 0: 0.27978, 5: 0.17592, 6: 0.25918},
 91: {4: 0, 3: 0, 0: 0.5195, 7: 0.16284},
 92: {0: 0, 1: 0, 5: 0.56873},
 93: {4: 0, 7: 0, 1: 0, 0: 0.35504, 5: 0.26948, 8: 0.1881},
 96: {1: 0, 6: 0, 3: 0.38494, 7: 0.27639},
 97: {0: 0, 3: 0.4569},
 98: {1: 0, 0: 0, 2: 0.00989, 4: 0.19691, 5: 0.42121, 7: 0.01611, 9: 0.11857},
 100: {5: 0, 6: 0, 0: 0.35529, 4: 0.21375, 7: 0.16701, 9: 0.08554},
 101: {0: 0, 2: 0, 1: 0.01742, 4: 0.17201, 6: 0.16303, 8: 0.02297, 9: 0.40074},
 103: {1: 0},
 108: {0: 0, 3: 0.17123, 6: 0.33405},
 113: {1: 0, 9: 0, 3: 0.52191, 6: 0.17636},
 116: {2: 0, 9: 0, 0: 0.4248, 3: 0.02407, 7: 0.24248, 8: 0.06192},
 119: {2: 0, 0: 0.40093, 6: 0.25618},
 120: {2: 0, 9: 0, 0: 0.44997, 3: 0.00657, 8: 0.24921},
 123: {4: 0, 0: 0.21157, 5: 0.13037, 6: 0.25444, 8: 0.09301},
 124: {3: 0, 8: 0, 6: 0.33304, 7: 0.19815, 9: 0.1772},
 125: {0: 0, 2: 0.02308, 4: 0.12656, 6: 0.32635, 7: 0.23576},
 128: {3: 0, 0: 0.13393, 4: 0.0058, 9: 0.29864},
 131: {7: 0, 3: 0, 0: 0.19032, 2: 0.03078, 4: 0.12547, 6: 0.41159}}
        self.shift_timers_dict = {'Components': {1: [6], 2: [9], 7: [3, 6], 15: [4, 3], 25: [1], 28: [3], 38: [4, 1], 43: [4, 0], 50: [0], 52: [0], 53: [1, 6], 56: [8], 63: [2], 69: [4], 73: [4, 6], 75: [3], 79: [1], 80: [1, 4, 6, 7, 8, 9], 90: [1, 2], 91: [4, 3], 92: [0, 1], 93: [4, 7, 1], 96: [1, 6], 97: [0], 98: [1, 0], 100: [5, 6], 101: [0, 2], 103: [1], 108: [0], 113: [1, 9], 116: [2, 9], 119: [2], 120: [2, 9], 123: [4], 124: [3, 8], 125: [0], 128: [3], 131: [7, 3]}, 
                                  'Targets': {'PGE': 0, 'SMUD': 0, 'SCE': 21, 'SDGE': 0, 'Portland':22, 'Utah':20}}
        self.weekend_timers_dict = {2: {9: 0, 7: 0, 3: 0.30036, 6: 0.05497, 8: 0.09777},
 15: {0: 0, 2: 0.05363, 4: 0.07807, 5: 0.16625, 8: 0.2602},
 25: {4: 0, 6: 0, 0: 0.30698, 5: 0.06866},
 38: {9: 0, 7: 0, 8: 0, 0: 0.1955, 6: 0.31529},
 43: {3: 0, 5: 0, 1: 0.21035, 7: 0.11394, 9: 0.12133},
 50: {1: 0, 2: 0.02924, 4: 0.13635, 7: 0.12723, 9: 0.19408},
 52: {1: 0, 2: 0.44318},
 91: {0: 0, 7: 0.16236, 8: 0.14845, 9: 0.2101},
 96: {0: 0, 4: 0.03199, 5: 0.08512, 6: 0.19454, 8: 0.15892, 9: 0.10285},
 116: {5: 0, 0: 0.19057, 3: 0.24956, 9: 0.15634},
 124: {1: 0, 5: 0.11576, 6: 0.20798, 8: 0.05629, 9: 0.18691},
 125: {6: 0, 1: 0.19651, 4: 0.2398},
 128: {0: 0, 5: 0.29316, 8: 0.12252}}
        self.weekend_shift_timers_dict = {'Components': {2: [9, 7], 15: [0], 25: [4, 6], 38: [9, 7, 8], 43: [3, 5], 50: [1], 52: [1], 91: [0], 96: [0], 116: [5], 124: [1], 125: [6], 128: [0]},
                                          'Targets': {'PGE': 0, 'SMUD': 0, 'SCE': 21, 'SDGE': 0, 'Portland':22, 'Utah':20}}
        self.timer_cat = 'home_l2'            
        
    def set_up_dendrogram_cluster_mapping(self):

        self.cluster_reorder_dendtoac = {i: i for i in range(self.ng)}
        self.cluster_reorder_actodend = {i: i for i in range(self.ng)}


class SPEECh(object):
    """
    Overall SPEECh class - where the differences between the CP and EVI-Pro calculations are stored.
    Calculations behind the distribution over driver groups happen here.

    :param data: Object of :class:`DataSetConfigurations'
    :type data: class:`DataSetConfigurations'
    """
    
    def __init__(self, data, penetration_level=0.5, run_weekend_pgs=False, outside_california=True, states=None, simple_adoption=True):

        self.data = data
        self.penetration_level = penetration_level
        self.run_weekend_pgs = run_weekend_pgs
        
        self.adoption_df = pd.read_csv(data.folder+'adoption_df_us_counties.csv')
        if states is not None:
            self.adoption_df = self.adoption_df[self.adoption_df['State'].isin(states)].copy(deep=True).reset_index(drop=True) # eg [CA, WA]
        self.set_up_adoption(simple_adoption)
        self.num_evs = None
        self.local_pen_level = None
        self.local_num_vehicles = None
        self.local_population = None

        self.pz = {'weekday': {}, 'weekend': {}}
        self.pg = None
        self.p_abe_data = None
        self.pg_abe_data = None
        self.pb_i_data = None
        self.pa_home_ih = None
        self.pa_work_i = None
        self.pa_ih_data = None
        self.pe_bd_data = None
        self.pdih_data = None

        self.regions_mapping = None
        self.regions_with_missing_data = []
        self.region_total_p_abe = None

        self.pz_g()

    def set_up_adoption(self, simple_adoption=True):
        """This method can be used to interpolate adoption in each location between current and total electrification rates. That functionality is not used in this paper as we simulate 100% adoption, so the the distribution just matches the distribution of all current light-duty vehicles."""
        
        pev_col = 'Current PEVs'
        veh_col = 'Current Vehicles'
            
        if self.penetration_level == 1.0:
            self.adoption_df['New Distribution'] = self.adoption_df['Goal Fraction']
            self.adoption_df['New Num Vehicles'] = self.adoption_df[veh_col].astype(int)
        elif simple_adoption:
            self.adoption_df['New Num Vehicles'] = (self.penetration_level * self.adoption_df[veh_col].values).astype(int)
            self.adoption_df['New Distribution'] = self.adoption_df['New Num Vehicles'] / self.adoption_df[veh_col].sum()
        else: #interpolate between current and final
            current_adoption_level = self.adoption_df[pev_col].sum() / self.adoption_df[veh_col].sum()
            total_veh = self.penetration_level * self.adoption_df[veh_col].sum()
            self.adoption_df['New Distribution'] = self.adoption_df['Current Fraction'] + ((self.adoption_df['Goal Fraction'].values - self.adoption_df['Current Fraction'].values)/(1-current_adoption_level)) * (self.penetration_level - current_adoption_level)

            inds1 = self.adoption_df[self.adoption_df['New Distribution'] * total_veh > self.adoption_df[veh_col]].index
            inds2 = self.adoption_df[self.adoption_df['New Distribution'] * total_veh <= self.adoption_df[veh_col]].index
            self.adoption_df.loc[inds1, 'New Num Vehicles'] = self.adoption_df.loc[inds1, veh_col]
            self.adoption_df.loc[inds2, 'New Num Vehicles'] = total_veh * self.adoption_df.loc[inds2, 'New Distribution']
            extra = (self.adoption_df.loc[inds1, 'New Distribution'] * total_veh - self.adoption_df.loc[inds1, 'New Num Vehicles']).sum()

            self.adoption_df['New Num Vehicles'] = self.adoption_df['New Num Vehicles'].astype(int)
            self.adoption_df['Additional'] = self.adoption_df['New Num Vehicles'].astype(int)

            while extra > 1:
                print('Redistributing extra EVs (% of state total): ', np.round(extra / total_veh, 3))
                self.adoption_df.loc[inds2, 'Additional'] += extra * self.adoption_df.loc[inds2, 'New Distribution']
                inds2 = self.adoption_df[np.ceil(self.adoption_df['Additional']) < self.adoption_df[veh_col]].index
                inds3 = self.adoption_df[np.ceil(self.adoption_df['Additional']) > self.adoption_df[veh_col]].index
                self.adoption_df.loc[inds3, 'New Num Vehicles'] = self.adoption_df.loc[inds3, veh_col].astype(int)
                self.adoption_df.loc[inds2, 'New Num Vehicles'] = self.adoption_df.loc[inds2, 'Additional'].astype(int)
                extra = (np.ceil(self.adoption_df.loc[inds3, 'Additional']) - self.adoption_df.loc[inds3, 'New Num Vehicles']).sum()
                self.adoption_df.loc[inds3, 'Additional'] = self.adoption_df.loc[inds3, veh_col].astype(int)

    def pg_(self):
        """A starting point (calculates pg) from saved p_g."""
        if self.run_weekend_pgs:
            self.pg = pd.read_csv(self.data.folder+'pg_we.csv')
        else:
            self.pg = pd.read_csv(self.data.folder+'pg.csv')
        
    def pz_g(self):
        
        for i in range(self.data.ng):
            self.pz['weekday'][i] = pd.read_csv(self.data.folder+'pz_weekday_g_'+str(i)+'.csv')
            if self.data.weekend_exists:
                if self.data.separate_weekend and (i in self.data.groups_with_separate_weekends):
                    self.pz['weekend'][i] = pd.read_csv(self.data.folder+'pz_weekend_g_'+str(i)+'_we.csv')
                else:
                    self.pz['weekend'][i] = pd.read_csv(self.data.folder+'pz_weekend_g_'+str(i)+'.csv')
            
    def pb(self):

        self.pb = pd.read_csv(self.data.folder+'pb.csv', index_col=0)

    def pe(self):

        self.pe = pd.read_csv(self.data.folder+'pe.csv', index_col=0)

    def p_abe_separate(self, a, b, e):
        # index must match keys in pg_abe file. E.g. work_0_home_0_largebattery_0_energy_0_1000
        # example inputs:
        # a = {'work_0_home_0':0.2, 'work_0_home_1':0.2, 'work_1_home_0':0.4, 'work_1_home_1':0.2}
        # b = {'largebattery_0':0.5, 'largebattery_1':0.5}
        # e = {'energy_0_1000':0.2, 'energy_1000_2000':0.2, 'energy_2000_3000':0.4, 'energy_3000_5000':0.2}

        a_inds = ['work_0_home_0', 'work_0_home_l1', 'work_0_home_l2', 'work_0_home_mud',
                  'work_paid_home_0', 'work_paid_home_l1', 'work_paid_home_l2', 'work_paid_home_mud',
                  'work_free_home_0', 'work_free_home_l1', 'work_free_home_l2', 'work_free_home_mud']
        b_inds = ['largebattery_0', 'largebattery_1']
        e_inds = ['energy_0_600', 'energy_600_1000', 'energy_1000_1600',
                  'energy_1600_2000', 'energy_2000_3000', 'energy_3000_4000',
                  'energy_4000_6000']
        all_inds = []
        for a1 in a_inds:
            for b1 in b_inds:
                for e1 in e_inds:
                    all_inds.append(a1+'_'+b1+'_'+e1)

        self.p_abe_data = pd.DataFrame(np.zeros((len(all_inds), )), index=all_inds, columns=['p_abe'])
        for key1, val1 in a.items():
            for key2, val2 in b.items():
                for key3, val3 in e.items():
                    self.p_abe_data.loc[key1+'_'+key2+'_'+key3] = val1*val2*val3

    def pg_abe(self):
        """A starting point (calculates pg) from saved or calculated p_abe."""

        if self.p_abe_data is None:
            self.p_abe_data = pd.read_csv(self.data.folder+'p_abe.csv', index_col=0)

        if self.run_weekend_pgs:
            self.pg_abe_data = pd.read_csv(self.data.folder+'pg_abe_workprice_we.csv')
        else:
            self.pg_abe_data = pd.read_csv(self.data.folder+'pg_abe_workprice.csv')
        self.pg = pd.DataFrame({'pg': np.zeros((self.data.ng,))})
        for key in self.p_abe_data.index.values:
            self.pg['pg'] += self.p_abe_data.loc[key, 'p_abe'] * self.pg_abe_data[key].values

    def pb_i(self, from_data=False, scenario='BaseCase'):
        """Scenarios to be based on survey data. Base case assumes large battery is more expensive than small.
        From data case is not really representative, since almost all high income."""
        if from_data:
            self.pb_i_data = pd.read_csv(self.data.folder+'pb_i.csv', index_col=0)
        else:
            if scenario == 'BaseCase':  # fit from CVRP data
                self.pb_i_data = pd.DataFrame({'I_0': [0.6941, 0.3059], 'I_1': [0.6675, 0.3325],
                                               'I_2': [0.6207, 0.3793]}, #refit with 0-60, 60-100, 100+
                                              index=['largebattery_0', 'largebattery_1'])
            elif scenario == 'Equal':
                self.pb_i_data = pd.DataFrame({'I_0': [0.0, 1.0], 'I_1': [0.0, 1.0],
                                               'I_2': [0.0, 1.0], 'I_3': [0.0, 1.0]},
                                              index=['largebattery_0', 'largebattery_1'])
            else:
                print('Unknown scenario.')

    def pa_ih(self, scenario='BaseCase'):
        """Only home depends on home type, work access just modeled on income."""
        if scenario == 'HighHome':
            """H_0 is "Other" mobile homes and boats, so no charging. H_1 is large MUD. H_2 is small MUD. 
            New:H_3 is attached house. H_4 is detached house. Old:houses renter occupied. H_4 is houses owner occupied."""
            self.pa_home_ih = pd.DataFrame({'I_0_H_0': [0.0, 0.0, 0.243, 0.757], 
                            'I_1_H_0': [0.0, 0.0, 0.231, 0.769],
                            'I_2_H_0': [0.0, 0.0, 0.354, 0.646], 
                            'I_0_H_1': [0.0, 0.0, 0.265, 0.735], 
                            'I_1_H_1': [0.0, 0.0, 0.191, 0.809],
                            'I_2_H_1': [0.0, 0.0, 0.443, 0.557], 
                            'I_0_H_2': [0.0, 0.0, 0.326, 0.674], 
                            'I_1_H_2': [0.0, 0.0, 0.474, 0.526],
                            'I_2_H_2': [0.0, 0.0, 0.528, 0.472], 
                            'I_0_H_3': [0.0, 0.647, 0.0, 0.353], 
                            'I_1_H_3': [0.0, 0.687, 0.0, 0.313], 
                            'I_2_H_3': [0.0, 0.778, 0.0, 0.222], 
                            'I_0_H_4': [0.0, 0.777, 0.0, 0.223], 
                            'I_1_H_4': [0.0, 0.824, 0.0, 0.176],
                            'I_2_H_4': [0.0, 0.873, 0.0, 0.127]}, 
                           index=['home_l1', 'home_l2', 'home_mud', 'home_0'])
            self.pa_work_i = pd.DataFrame({'I_0': [0.25*0.1, 0.75*0.1, 0.9], 'I_1': [0.25*0.3, 0.75*0.3, 0.7], 
                                           'I_2': [0.25*0.5, 0.75*0.5, 0.5]},
                                          index=['work_paid', 'work_free', 'work_0'])
        elif scenario == 'UniversalHome':
            self.pa_home_ih = pd.DataFrame({'I_0_H_0': [0.0, 0.0, 1.0, 0.0], 'I_1_H_0': [0.0, 0.0, 1.0, 0.0],
                                            'I_2_H_0': [0.0, 0.0, 1.0, 0.0],
                                            'I_0_H_1': [0.0, 0.0, 1.0, 0.0], 'I_1_H_1': [0.0, 0.0, 1.0, 0.0],
                                            'I_2_H_1': [0.0, 0.0, 1.0, 0.0],
                                            'I_0_H_2': [0.0, 0.0, 1.0, 0.0], 'I_1_H_2': [0.0, 0.0, 1.0, 0.0],
                                            'I_2_H_2': [0.0, 0.0, 1.0, 0.0],
                                            'I_0_H_3': [0.0, 1.0, 0.0, 0.0], 'I_1_H_3': [0.0, 1.0, 0.0, 0.0],
                                            'I_2_H_3': [0.0, 1.0, 0.0, 0.0],
                                            'I_0_H_4': [0.0, 1.0, 0.0, 0.0], 'I_1_H_4': [0.0, 1.0, 0.0, 0.0],
                                            'I_2_H_4': [0.0, 1.0, 0.0, 0.0]},
                                           index=['home_l1', 'home_l2', 'home_mud', 'home_0'])
            self.pa_work_i = pd.DataFrame({'I_0': [0.25*0.1, 0.75*0.1, 0.9], 'I_1': [0.25*0.3, 0.75*0.3, 0.7], 
                                           'I_2': [0.25*0.5, 0.75*0.5, 0.5]},
                                          index=['work_paid', 'work_free', 'work_0'])
        elif scenario == 'LowHome_LowWork':
            """Cut rates of home access relative to base case.
            H_0 is "Other" mobile homes and boats, so no charging. H_1 is large MUD. H_2 is small MUD. 
            New:H_3 is attached house. H_4 is detached house. Old:houses renter occupied. H_4 is houses owner occupied."""
            self.pa_home_ih = pd.DataFrame({'I_0_H_0': [0.0, 0.0, 0.097, 0.903], 
                            'I_1_H_0': [0.0, 0.0, 0.089, 0.911],
                            'I_2_H_0': [0.0, 0.0, 0.091, 0.909], 
                            'I_0_H_1': [0.0, 0.0, 0.104, 0.896], 
                            'I_1_H_1': [0.0, 0.0, 0.083, 0.917],
                            'I_2_H_1': [0.0, 0.0, 0.259, 0.741], 
                            'I_0_H_2': [0.0, 0.0, 0.094, 0.906], 
                            'I_1_H_2': [0.0, 0.0, 0.288, 0.712],
                            'I_2_H_2': [0.0, 0.0, 0.253, 0.747], 
                            'I_0_H_3': [0.0, 0.220, 0.0, 0.780], 
                            'I_1_H_3': [0.0, 0.279, 0.0, 0.721], 
                            'I_2_H_3': [0.0, 0.329, 0.0, 0.671], 
                            'I_0_H_4': [0.0, 0.280, 0.0, 0.720], 
                            'I_1_H_4': [0.0, 0.297, 0.0, 0.703],
                            'I_2_H_4': [0.0, 0.335, 0.0, 0.665]}, 
                           index=['home_l1', 'home_l2', 'home_mud', 'home_0'])
            self.pa_work_i = pd.DataFrame({'I_0': [0.25*0.1, 0.75*0.1, 0.9], 'I_1': [0.25*0.3, 0.75*0.3, 0.7], 
                                           'I_2': [0.25*0.5, 0.75*0.5, 0.5]},
                                          index=['work_paid', 'work_free', 'work_0'])

        elif scenario == 'LowHome_HighWork':
            """Cut rates of home access relative to base case.
            H_0 is "Other" mobile homes and boats, so no charging. H_1 is large MUD. H_2 is small MUD. 
            New:H_3 is attached house. H_4 is detached house. Old:houses renter occupied. H_4 is houses owner occupied."""
            self.pa_home_ih = pd.DataFrame({'I_0_H_0': [0.0, 0.0, 0.097, 0.903], 
                            'I_1_H_0': [0.0, 0.0, 0.089, 0.911],
                            'I_2_H_0': [0.0, 0.0, 0.091, 0.909], 
                            'I_0_H_1': [0.0, 0.0, 0.104, 0.896], 
                            'I_1_H_1': [0.0, 0.0, 0.083, 0.917],
                            'I_2_H_1': [0.0, 0.0, 0.259, 0.741], 
                            'I_0_H_2': [0.0, 0.0, 0.094, 0.906], 
                            'I_1_H_2': [0.0, 0.0, 0.288, 0.712],
                            'I_2_H_2': [0.0, 0.0, 0.253, 0.747], 
                            'I_0_H_3': [0.0, 0.220, 0.0, 0.780], 
                            'I_1_H_3': [0.0, 0.279, 0.0, 0.721], 
                            'I_2_H_3': [0.0, 0.329, 0.0, 0.671], 
                            'I_0_H_4': [0.0, 0.280, 0.0, 0.720], 
                            'I_1_H_4': [0.0, 0.297, 0.0, 0.703],
                            'I_2_H_4': [0.0, 0.335, 0.0, 0.665]}, 
                           index=['home_l1', 'home_l2', 'home_mud', 'home_0'])
            # New: 60%, 60%, 80%, 100% (same rate of paid:free)
            self.pa_work_i = pd.DataFrame({'I_0': [0.25*0.4, 0.75*0.4, 1-0.4], 'I_1': [0.25*0.5, 0.75*0.5, 1-0.5], 
                                           'I_2': [0.25*0.55, 0.75*0.55, 1-0.55]}, 
                                          index=['work_paid', 'work_free', 'work_0'])

        self.pa_ih_data = pd.DataFrame(np.zeros((int(len(self.pa_home_ih)*len(self.pa_work_i)), len(self.pa_home_ih.columns))),
                                       columns=self.pa_home_ih.columns,
                                       index=[key2+'_'+key1 for key1 in self.pa_home_ih.index.values for key2 in self.pa_work_i.index.values])
        workkeys = ['work_paid', 'work_free', 'work_0']
        for key1 in ['home_l1', 'home_l2', 'home_mud', 'home_0']:
            for key2 in workkeys:
                for col in self.pa_work_i.columns:
                    for hkey in ['H_'+str(i) for i in [0, 1, 2, 3, 4]]:
                        self.pa_ih_data.loc[key2+'_'+key1, col+'_'+hkey] = self.pa_work_i.loc[key2, col] * self.pa_home_ih.loc[key1, col+'_'+hkey]

    def pe_bd(self):

        self.pe_bd_data = pd.read_csv(self.data.folder + 'pe_d.csv', index_col=0)

    def pg_dih(self, region_type, region_value, return_abe=False):
        """For a particular entry in the dih data, calculate the chain."""

        if self.pdih_data is None:
            self.pdih_data = pd.read_csv(self.data.folder + 'pdih_us_counties.csv', index_col=0)
        
        if self.pb_i_data is None:
            self.pb_i()
        if self.pe_bd_data is None:
            self.pe_bd()
        if self.pa_ih_data is None:
            self.pa_ih()

        local_df = self.pdih_data.loc[self.pdih_data[region_type] == region_value]
        local_df = local_df.rename(columns={'P_LowIncome': 'I_0', 'P_MedIncome': 'I_1', 'P_HighIncome': 'I_2'})
        pb = 0*self.pb_i_data[self.pb_i_data.columns[0]]
        i_inds = ['I_'+str(i) for i in range(3)]
        for i in i_inds:
            pb += local_df[i].values[0] * self.pb_i_data[i]
        b_inds = pb.index.values
        d_inds = ['D_'+str(i) for i in range(11)]
        pe = 0*self.pe_bd_data[self.pe_bd_data.columns[0]]
        for j in range(len(d_inds)):
            pe += (self.pe_bd_data[d_inds[j]] * local_df[d_inds[j]].values[0])

        local_df['H_0'] = local_df['P_Other']
        local_df['H_1'] = local_df['P_LargeApt']
        local_df['H_2'] = local_df['P_SmallApt']
        local_df['H_3'] = local_df['P_House_Attached']
        local_df['H_4'] = local_df['P_House_Detached']

        h_inds = ['H_'+str(i) for i in range(5)]
        pa = 0*self.pa_ih_data[self.pa_ih_data.columns[0]]
        for i in range(len(i_inds)):
            for j in range(len(h_inds)):
                pa += local_df[i_inds[i]].values[0] * self.pa_ih_data['I_'+str(int(i))+'_'+h_inds[j]] * local_df[h_inds[j]].values[0]

        self.p_abe_separate(dict(pa), dict(pb), dict(pe))
        self.pg_abe()

        if return_abe:
            return self.pg, pa, pb, pe
        else:
            return self.pg

    def num_evs_region(self, zipcode):
        idx = self.adoption_df[self.adoption_df['County FIPS'] == zipcode].index
        if len(idx) > 0:
            return int(self.adoption_df.loc[idx, 'New Num Vehicles'].values[0])
        else:
            print('Missing Num EVs in ', zipcode)
            self.regions_with_missing_data.append(zipcode)
            return 0

    def num_vehs_region(self, zipcode):
        idx = self.adoption_df[self.adoption_df['County FIPS'] == zipcode].index
        if len(idx) > 0:
            return int(self.adoption_df.loc[idx, 'Current Vehicles'].values[0])
        else:
            print('Missing Num Vehs in ', zipcode)
            return 0

    def pg_multiple_regions(self, region_type, region_value_list, pop_weight=False, return_pabe=False):

        total_pg = pd.DataFrame({'pg': np.zeros((self.data.ng,))})
        total_pabe = None
        total_pop = 0
        total_people_pop = 0
        total_veh = 0
        if self.pdih_data is None:
            self.pdih_data = pd.read_csv(self.data.folder + 'pdih_us_counties.csv', index_col=0)
            
        subset = self.pdih_data.loc[self.pdih_data[region_type].isin(region_value_list)]
        for i in subset.index:
            region_pg = self.pg_dih('FIPS', subset.loc[i, 'FIPS'])
            if region_pg['pg'].sum() < 0.99:
                print('Region PG < 0.99: ', subset.loc[i, 'FIPS'])
                if region_pg['pg'].sum() < 0.9:
                    print('Worse, region PG < 0.9')
            if pop_weight:
                pop = subset.loc[i, '# Total Population, 2019 [Estimated]']
            else:
                pop = self.num_evs_region(subset.loc[i, 'FIPS'])
                veh = self.num_vehs_region(subset.loc[i, 'FIPS'])
                people_pop = subset.loc[i, '# Total Population, 2019 [Estimated]']
                total_people_pop += people_pop
            total_pg += pop * region_pg
            total_pop += pop
            total_veh += veh
            if total_pabe is None:
                total_pabe = pop * self.p_abe_data.copy(deep=True)
            else:
                total_pabe += pop * self.p_abe_data.copy(deep=True)
        total_pg = total_pg / total_pop
        total_pabe = total_pabe / total_pop

        if not pop_weight:
            self.num_evs = total_pop
            self.local_pen_level = total_pop / total_veh
            self.local_num_vehicles = total_veh
            self.local_population = total_people_pop

        self.pg = total_pg.copy(deep=True)
        self.region_total_p_abe = total_pabe

        if len(self.regions_with_missing_data) > 0:
            print(str(len(self.regions_with_missing_data))+' zips with missing data.')

        if return_pabe:
            return total_pabe


class SPEEChGeneralConfiguration(object):
    """General configuration using the speech class to set up the driver groups and segments."""
    def __init__(self, speech, remove_timers=False, utility_region='PGE'):
        
        self.speech = speech
        if self.speech.pg['pg'].sum() < 0.99:
            print('sum(PG) < 0.99')
        self.group_configs = {}
        self.num_drivers = np.zeros((self.speech.data.ng, ))
        self.time_step = (1/60)
        self.num_time_steps = 1440
        self.time_steps_per_hour = 60
        self.energy_clip = 100
        self.num_total_drivers = None
        self.total_load_dict = {}
        self.total_load_segments = np.zeros((self.num_time_steps, len(self.speech.data.labels)))
        self.all_load_dicts = {}
        self.all_load_segments = {}

        self.remove_timers = remove_timers
        self.utility_region = utility_region
        self.shift_timers = False
        if utility_region != 'PGE':
            self.shift_timers = True

        if self.speech.num_evs is not None:
            self.num_evs(self.speech.num_evs)
            self.groups()
        
    def num_evs(self, num_total_drivers, col='pg'):
        """Calculate the number of EVs in each driver group."""

        self.num_total_drivers = num_total_drivers
        for i in range(self.speech.data.ng):
            self.num_drivers[i] = int(self.num_total_drivers * self.speech.pg.loc[i, col])
        
    def groups(self):
        """Load the model files for each driver group."""
        for i in range(self.speech.data.ng):
            self.group_configs[i] = SPEEChGroupConfiguration(self, i)
            if self.remove_timers:
                if (i in self.speech.data.timers_dict.keys()) and (self.num_drivers[i] > 0):
                    self.change_ps_zg(i, self.speech.data.timer_cat, 'weekday', self.speech.data.timers_dict[i])
                if (self.speech.data.weekend_timers_dict is not None) and (i in self.speech.data.weekend_timers_dict.keys()):
                    self.change_ps_zg(i, self.speech.data.timer_cat, 'weekend', self.speech.data.weekend_timers_dict[i])

    def change_ps_zg(self, g, cat, weekday, new_weights):
        """Reweight components in a group's sessions mixture model, for example to remove timers."""
        gmm = self.group_configs[g].segment_gmms[weekday][cat]
        
        all_inds = np.arange(0, np.shape(gmm.weights_)[0])
        total_rem = 1
        for key, val in new_weights.items():
            gmm.weights_[key] = val
            total_rem -= val
        other_keys = np.delete(all_inds, list(new_weights.keys()))
        gmm.weights_[other_keys] = gmm.weights_[other_keys] * total_rem / sum(gmm.weights_[other_keys])
        
        self.group_configs[g].segment_gmms[weekday][cat] = gmm

    def change_pg_dend(self, new_weights):

        new_dict = {}
        for key, val in new_weights.items():
            new_dict[self.speech.data.cluster_reorder_dendtoac[key]] = val

        self.change_pg(new_dict)

    def change_pg(self, new_weights, new_col='pg'):
        """Reweight driver groups."""

        self.speech.pg[new_col] = self.speech.pg['pg'].copy()
        all_inds = np.arange(0, self.speech.data.ng)
        total_rem = 1
        for key, val in new_weights.items():
            self.speech.pg.loc[key, new_col] = val
            total_rem -= val
        other_keys = np.delete(all_inds, list(new_weights.keys()))
        self.speech.pg.loc[other_keys, new_col] = self.speech.pg.loc[other_keys, new_col] * total_rem / sum(self.speech.pg.loc[other_keys, new_col])
        
    def run_all(self, verbose=False, weekday='weekday'):
        """Run model and calculate load profiles for each group."""
        
        self.total_load_dict = {x: np.zeros((self.num_time_steps,)) for x in self.speech.data.labels}
        self.total_load_segments = np.zeros((self.num_time_steps, len(self.speech.data.labels)))
        self.all_load_dicts = {}
        self.all_load_segments = {}
        for g in range(self.speech.data.ng):
            if verbose:
                print('Group '+str(g))
            model = LoadProfile(self, self.group_configs[g], weekday=weekday)
            model.calculate_load()
            self.all_load_dicts[g] = model.load_segments_dict
            self.all_load_segments[g] = model.load_segments_array
            for key, val in model.load_segments_dict.items():
                self.total_load_dict[key] += val
            self.total_load_segments += model.load_segments_array

        
class SPEEChGroupConfiguration(object):
    """Configuration, sessions counts, and gmms for each individual driver group. """
    
    def __init__(self, speech_config, g):
        
        self.speech_config = speech_config
        self.g = g
        self.total_drivers = speech_config.num_drivers[g]
        self.segment_session_numbers = {'weekday': {}, 'weekend': {}}
        self.segment_gmms = {'weekday': {}, 'weekend': {}}
        
        self.numbers()
        self.load_gmms()

    def numbers(self, total_drivers=None):
        """Calculate the number of sessions to simulate in each charging segment for this group."""
        
        if total_drivers is not None:
            self.total_drivers = total_drivers
        
        inds = np.random.choice(range(len(self.speech_config.speech.pz['weekday'][self.g])), int(self.total_drivers), replace=True)
        for cat in self.speech_config.speech.data.categories:
            if os.path.isfile(self.speech_config.speech.data.folder+'GMMs/'+self.speech_config.speech.data.gmm_subfolder+'weekday'+'_'+self.speech_config.speech.data.gmm_names[cat]+'_'+str(self.g)+'.p'):
                self.segment_session_numbers['weekday'][cat] = int(sum(self.speech_config.speech.pz['weekday'][self.g].loc[inds, cat+self.speech_config.speech.data.zkey_weekday]))
            else:
                self.segment_session_numbers['weekday'][cat] = 0
        if self.speech_config.speech.data.weekend_exists:
            inds = np.random.choice(range(len(self.speech_config.speech.pz['weekend'][self.g])), int(self.total_drivers), replace=True)
            for cat in self.speech_config.speech.data.categories:
                key1 = self.speech_config.speech.data.folder+'GMMs/'+self.speech_config.speech.data.gmm_subfolder+'weekend'+'_'+self.speech_config.speech.data.gmm_names[cat]+'_'+str(self.g)+'.p'
                if os.path.isfile(key1):
                    self.segment_session_numbers['weekend'][cat] = int(sum(self.speech_config.speech.pz['weekend'][self.g].loc[inds, cat+self.speech_config.speech.data.zkey_weekend]))
                else:
                    self.segment_session_numbers['weekend'][cat] = 0

    def load_gmms(self):
        """ Load the GMM files for this group."""

        for cat in self.speech_config.speech.data.categories:
            weekday = 'weekday'
            key = self.speech_config.speech.data.folder+'GMMs/'+self.speech_config.speech.data.gmm_subfolder+weekday+'_'+self.speech_config.speech.data.gmm_names[cat]+'_'+str(self.g)+'.p'
            if os.path.isfile(key):
                self.segment_gmms[weekday][cat] = pickle.load(open(key, "rb"))
            if self.speech_config.speech.data.weekend_exists:
                weekday = 'weekend'
                key = self.speech_config.speech.data.folder+'GMMs/'+self.speech_config.speech.data.gmm_subfolder+weekday+'_'+self.speech_config.speech.data.gmm_names[cat]+'_'+str(self.g)+'.p'
                if os.path.isfile(key):
                    self.segment_gmms[weekday][cat] = pickle.load(open(key, "rb"))


class LoadProfile(object):
    """Calculating the load profile from the gmms and configurations."""
    
    def __init__(self, config, group_config, weekday='weekday'):
        """Other option for weekday: 'weekend'."""

        self.config = config  # speech general configuration
        self.group_config = group_config
        self.weekday = weekday
        self.load_segments_dict = {}
        self.load_segments_array = np.zeros((self.config.num_time_steps, self.config.speech.data.num_categories))
        self.energies = {}
        self.start_times = {}
        self.durations = {}
        self.gmm_outputs = {}

    def calculate_load(self):
        """Calculate the load profiles for each segment."""

        for segment_number in range(self.config.speech.data.num_categories):
            cat = self.config.speech.data.categories[segment_number]
            num_vehicles = self.group_config.segment_session_numbers[self.weekday][cat]
            if (num_vehicles > 0) and (cat in self.group_config.segment_gmms[self.weekday].keys()):
                gmm = self.group_config.segment_gmms[self.weekday][cat]
                if gmm.n_components > 1:
                    full_output = gmm.sample(num_vehicles)
                    if (self.weekday == 'weekend') and (self.config.speech.data.weekend_shift_timers_dict is not None):
                        if self.config.shift_timers & (cat == self.config.speech.data.timer_cat) & (self.group_config.g in self.config.speech.data.weekend_shift_timers_dict['Components'].keys()):
                            output = self.shift_timers(full_output)
                        else:
                            output = full_output[0]
                    else:
                        if self.config.shift_timers & (cat == self.config.speech.data.timer_cat) & (self.group_config.g in self.config.speech.data.shift_timers_dict['Components'].keys()):
                            output = self.shift_timers(full_output)
                        else:
                            output = full_output[0]
                    output = output[np.random.choice(np.shape(output)[0], np.shape(output)[0], replace=False), :]
                    if self.config.speech.data.start_mod == 1:
                        start_times = (self.config.speech.data.start_time_scaler * np.mod(24*3600*output[:, 0], 24*3600)).astype(int)
                    else:
                        start_times = (self.config.speech.data.start_time_scaler * np.mod(output[:, 0], 24*3600)).astype(int)
                    energies = np.clip(np.abs(output[:, 1]), 0, self.config.energy_clip)
                    end_times, load, lengths = self.end_times_and_load(start_times, energies, self.config.speech.data.rates[segment_number], return_lengths=True)

                    self.energies[cat] = energies
                    self.start_times[cat] = start_times
                    self.durations[cat] = lengths
                    self.gmm_outputs[cat] = output
                else:
                    load = np.zeros((self.config.num_time_steps, ))
            else:
                load = np.zeros((self.config.num_time_steps, ))
            self.load_segments_dict[self.config.speech.data.labels[segment_number]] = load
            self.load_segments_array[:, segment_number] = load
            

    def shift_timers(self, output):
        """Shift timers to new time, e.g. midnight."""

        if (self.weekday == 'weekend') and (self.config.speech.data.weekend_shift_timers_dict is not None):
            comps = self.config.speech.data.weekend_shift_timers_dict['Components'][self.group_config.g]
            if self.config.utility_region == 'Mixed':
                target = None
            else:
                target = self.config.speech.data.weekend_shift_timers_dict['Targets'][self.config.utility_region]
        else:
            comps = self.config.speech.data.shift_timers_dict['Components'][self.group_config.g]
            if self.config.utility_region == 'Mixed':
                target = None
            else:
                target = self.config.speech.data.shift_timers_dict['Targets'][self.config.utility_region]
            
        for comp in comps:
            if self.config.speech.data.start_mod == 1:
                if target is None:
                    num_targets = np.shape(output[0][np.where(output[1] == comp)[0], 0])[0]
                    target_here = np.random.choice(np.array([20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 0, 0.5, 1, 1.5, 2, 2.5]), num_targets, replace=True)
                    output[0][np.where(output[1] == comp)[0], 0] = (1/24)*target_here
                else:
                    output[0][np.where(output[1] == comp)[0], 0] = (1/24)*target
            else:
                if target is None:
                    num_targets = np.shape(output[0][np.where(output[1] == comp)[0], 0])[0]
                    target_here = np.random.choice(np.array([20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 0, 0.5, 1, 1.5, 2, 2.5]), num_targets, replace=True)
                    output[0][np.where(output[1] == comp)[0], 0] = 3600*target_here
                else:
                    output[0][np.where(output[1] == comp)[0], 0] = 3600*target
        return output[0]

    def end_times_and_load(self, start_times, energies, rate, return_lengths=False):
        """From the start times, energies, and rates output by the sessions model, calculate the load profiles."""
        
        time_steps_per_hour = self.config.time_steps_per_hour
        num_time_steps = self.config.num_time_steps
        load = np.zeros((num_time_steps,))
        end_times = np.zeros(np.shape(start_times)).astype(int)

        lengths = (time_steps_per_hour * energies / np.abs(rate)).astype(int)
        extra_charges = energies - lengths * np.abs(rate) / time_steps_per_hour
        inds1 = np.where((start_times + lengths) > num_time_steps)[0]
        inds2 = np.delete(np.arange(0, np.shape(end_times)[0]), inds1)

        end_times[inds1] = (np.minimum(start_times[inds1].astype(int)+lengths[inds1]-num_time_steps, num_time_steps)).astype(int)
        end_times[inds2] = (start_times[inds2] + lengths[inds2]).astype(int)
        inds3 = np.where(end_times >= num_time_steps)[0]
        inds4 = np.delete(np.arange(0, np.shape(end_times)[0]), inds3)

        for i in range(len(inds1)):
            idx = int(inds1[i])
            load[np.arange(int(start_times[idx]), num_time_steps)] += rate * np.ones((num_time_steps - int(start_times[idx]),))
            load[np.arange(0, end_times[idx])] += rate * np.ones((end_times[idx],))
        for i in range(len(inds2)):
            idx = int(inds2[i])
            load[np.arange(int(start_times[idx]), end_times[idx])] += rate * np.ones((lengths[idx],))
        load[0] += np.sum(extra_charges[inds3] * time_steps_per_hour)
        for i in range(len(inds4)):
            load[end_times[int(inds4[i])]] += extra_charges[int(inds4[i])] * time_steps_per_hour

        if return_lengths:
            return end_times, load, lengths
        else:
            return end_times, load

        
class Plotting(object):
    """Plotting class - including final results of load profile, intermediate profiles, and distributions."""
    
    def __init__(self, speech, config=None, n=5e6):
        
        self.speech = speech
        if config is None:
            self.config = SPEEChGeneralConfiguration(speech)
            self.config.num_evs(n)
            self.config.groups()
        else:
            self.config = config
        
    def pg(self, col='pg'):
        """Plots a bar chart of P(G)."""
        vals = np.zeros((self.speech.data.ng, ))
        for i in range(self.speech.data.ng):
            j = self.speech.data.cluster_reorder_dendtoac[i]
            vals[i] = self.speech.pg.loc[j, col]
        plt.figure()
        plt.bar(np.arange(1, self.speech.data.ng+1), vals)
        plt.ylabel('P(G)')
        plt.xlabel('G')
        plt.show()
        
    def total(self, verbose=False, weekday='weekday', save_str=None):
        """Runs the model for a single day and plots the result."""
        self.config.run_all(verbose=verbose, weekday=weekday)
        self.plot_single(self.config.total_load_segments, self.config.total_load_dict, save_str=save_str)
    
    def groups(self, n=1e5, weekday='weekday', savestr=None, ncol=4):
        """Plots a sample profile for all of the groups."""
        nrow = int(np.ceil(np.divide(self.speech.data.ng, ncol)))
        fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(int(ncol*3), int(nrow*3)))
        config = copy.deepcopy(self.config)
        ymax = 0
        for i in range(self.speech.data.ng):
            row = int(np.divide(i, ncol))
            col = np.mod(i, ncol)
            j = self.speech.data.cluster_reorder_dendtoac[i]
            config.group_configs[j].numbers(total_drivers=n)
            model = LoadProfile(config, config.group_configs[j], weekday=weekday)
            model.calculate_load()
            if np.max(np.sum(model.load_segments_array, axis=1)) > ymax:
                ymax = np.max(np.sum(model.load_segments_array, axis=1))
            ylab = False; xlab=False; legend=False
            if col == 0:
                ylab = True
                if row == 0:
                    legend = True
            if row == (nrow-1):
                xlab = True
            self.plot(axes[row, col], model.load_segments_array, model.load_segments_dict, title='Group '+str(i+1), ylab=ylab, xlab=xlab, legend=legend)
        if (row == nrow) & (col < (ncol-1)):
            for col_left in np.arange(col, ncol):
                axes[row, col_left].set_axis_off()
        for i in range(nrow):
            for j in range(ncol):
                axes[row, col].set_ylim([0, (1/1000)*ymax])
        plt.tight_layout()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')
        plt.show()
                
    def sessions_components(self, g, cat, weekday, n=1e5):
        """Plots the components of the sessions mixture model for a particular group and segment."""
        gmm = self.config.group_configs[g].segment_gmms[weekday][cat]
        output = gmm.sample(n)
        output_values = output[0]
        output_labels = output[1]
        
        fig, ax = plt.subplots(1,1,figsize=(8,5))
        inds = np.arange(0, np.shape(output_values)[0])
        if self.speech.data.start_mod == 1:
            start_times = (self.speech.data.start_time_scaler * np.mod(24*3600*output_values[inds, 0], 24*3600)).astype(int)
        else:
            start_times = (self.speech.data.start_time_scaler * np.mod(output_values[inds, 0], 24*3600)).astype(int)
        energies = np.clip(np.abs(output_values[inds, 1]), 0, self.config.energy_clip)
        segment_number = np.where(np.array(self.speech.data.categories)==cat)[0][0]
        end_times, load = self.end_times_and_load(start_times, energies, self.speech.data.rates[segment_number])
        load_segments_array = np.zeros((self.config.num_time_steps, self.speech.data.num_categories))
        load_segments_array[:, segment_number] = load
        load_segments_dict = {self.speech.data.labels[segment_number]:load}
        self.plot(ax, load_segments_array, load_segments_dict, 'Total')
        plt.tight_layout()
        plt.show()
        
        nc = gmm.n_components
        fig, axes = plt.subplots(1, nc, sharex=True, sharey=True, figsize=(3*nc, 3))
        ymax = 0
        for i in range(nc):
            inds = np.where(output_labels == i)[0]
            if self.speech.data.tart_mod == 1:
                start_times = (self.speech.data.start_time_scaler * np.mod(24*3600*output_values[inds, 0], 24*3600)).astype(int)
            else:
                start_times = (self.speech.data.start_time_scaler * np.mod(output_values[inds, 0], 24*3600)).astype(int)
            energies = np.clip(np.abs(output_values[inds, 1]), 0, self.config.energy_clip)
            segment_number = np.where(np.array(self.speech.data.categories)==cat)[0][0]
            end_times, load = self.end_times_and_load(start_times, energies, self.speech.data.rates[segment_number])
            load_segments_array = np.zeros((self.config.num_time_steps, self.speech.data.num_categories))
            load_segments_array[:, segment_number] = load
            if np.max(load) > ymax:
                ymax = np.max(load)
            load_segments_dict = {self.speech.data.labels[segment_number]:load}
            self.plot(axes[i], load_segments_array, load_segments_dict, 'Weight: '+str(np.round(gmm.weights_[i], 2)))
        for i in range(nc):
            axes[i].set_ylim([0, (1/1000)*ymax])
        plt.tight_layout()
        plt.show()

    def plot(self, ax, load_segments_array, load_segments_dict, title, ylab=False, xlab=True, legend=False):
        """Standard plotting method."""
        x = (1/60)*np.arange(0, 1440)
        mark = np.zeros(np.shape(x))
        scaling = 1 / 1000
        unit = 'MW'
        if np.max(scaling * load_segments_array) > 1000:
            scaling = (1 / 1000) * (1 / 1000)
            unit = 'GW'
        for key, val in load_segments_dict.items():
            ax.plot(x, scaling * (mark + val), color=self.speech.data.colours[key])
            ax.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
            mark += val
        ax.plot(x, scaling * mark, 'k')
        ax.set_xlim([0, np.max(x)])
        if ylab:
            ax.set_ylabel(unit, fontsize=14)
        if xlab:
            ax.set_xlabel('Hour', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(bottom=0)
        if legend:
            ax.legend(loc='upper left', fontsize=12)
        
    def end_times_and_load(self, start_times, energies, rate):
        """Same calculation as in LoadProfile."""
        time_steps_per_hour = self.config.time_steps_per_hour
        num_time_steps = self.config.num_time_steps
        load = np.zeros((num_time_steps,))
        end_times = np.zeros(np.shape(start_times)).astype(int)

        lengths = (time_steps_per_hour * energies / rate).astype(int)
        extra_charges = energies - lengths * rate / time_steps_per_hour
        inds1 = np.where((start_times + lengths) > num_time_steps)[0]
        inds2 = np.delete(np.arange(0, np.shape(end_times)[0]), inds1)

        end_times[inds1] = (np.minimum(start_times[inds1].astype(int)+lengths[inds1]-num_time_steps, num_time_steps)).astype(int)
        end_times[inds2] = (start_times[inds2] + lengths[inds2]).astype(int)
        inds3 = np.where(end_times >= num_time_steps)[0]
        inds4 = np.delete(np.arange(0, np.shape(end_times)[0]), inds3)

        for i in range(len(inds1)):
            idx = int(inds1[i])
            load[np.arange(int(start_times[idx]), num_time_steps)] += rate * np.ones((num_time_steps - int(start_times[idx]),))
            load[np.arange(0, end_times[idx])] += rate * np.ones((end_times[idx],))
        for i in range(len(inds2)):
            idx = int(inds2[i])
            load[np.arange(int(start_times[idx]), end_times[idx])] += rate * np.ones((lengths[idx],))
        load[0] += np.sum(extra_charges[inds3] * time_steps_per_hour)
        for i in range(len(inds4)):
            load[end_times[int(inds4[i])]] += extra_charges[int(inds4[i])] * time_steps_per_hour

        return end_times, load

    def plot_single(self, load_segments_array, load_segments_dict, legend_subset=None, set_ylim=None, save_str=None, title=None):
        """Plot a single day profile."""
        x = (1/60)*np.arange(0, 1440)
        mark = np.zeros(np.shape(x))
        scaling = 1 / 1000
        unit = 'MW'
        if np.max(scaling * np.sum(load_segments_array, axis=1)) > 1000:
            scaling = (1 / 1000) * (1 / 1000)
            unit = 'GW'
        plt.figure(figsize=(8, 5))
        for key, val in load_segments_dict.items():
            plt.plot(x, scaling * (mark + val), color=self.speech.data.colours[key])
            if legend_subset is not None:
                if key in legend_subset:
                    plt.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
                else:
                    plt.fill_between(x, scaling * mark, scaling * (mark + val), color=self.speech.data.colours[key])
            else:
                plt.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
            mark += val
        plt.plot(x, scaling * mark, 'k')
        plt.legend(fontsize=12, loc='upper left')
        plt.xlim([0, np.max(x)])
        if set_ylim is None:
            plt.ylim([0, 1.1 * np.max(scaling * mark)])
        else:
            plt.ylim([0, set_ylim])
        plt.ylabel(unit, fontsize=14)
        plt.xlabel('Hour', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if title is not None:
            plt.set_title(title)
        if save_str is not None:
            plt.tight_layout()
            plt.savefig(save_str, bbox_inches='tight')
        plt.show()

    def plot_single_above_base(self, load_segments_array, load_segments_dict, base_load, legend_subset=None, set_ylim=None, save_str=None, title=None):
        """Plot EV demand above a given baseline demand."""
        x = (1/60)*np.arange(0, 1440)
        if np.shape(base_load)[0] == 24:
            xold = np.arange(0, 25)
            yold = np.zeros((25,))
            yold[np.arange(0, 24)] = np.copy(base_load)
            yold[-1] = yold[0]
            f2 = interp1d(xold, yold, kind='cubic')
            base_load = f2(x)
        elif np.shape(base_load)[0] == 288:
            xold = (1/12)*np.arange(0, 289)
            yold = np.zeros((289,))
            yold[np.arange(0, 288)] = np.copy(base_load)
            yold[-1] = yold[0]
            f2 = interp1d(xold, yold, kind='cubic')
            base_load = f2(x)

        # mark = np.zeros(np.shape(x))
        scaling = 1 / 1000
        unit = 'MW'
        if np.max(scaling * (base_load + np.sum(load_segments_array, axis=1))) > 1000:
            scaling = (1 / 1000) * (1 / 1000)
            unit = 'GW'
        plt.figure(figsize=(8, 5))
        plt.plot(x, scaling*base_load, color='k', alpha=0.8)
        plt.fill_between(x, 0, scaling*base_load, color='k', alpha=0.4, label='Base Load')
        mark = base_load
        for key, val in load_segments_dict.items():
            plt.plot(x, scaling * (mark + val), color=self.speech.data.colours[key])
            if legend_subset is not None:
                if key in legend_subset:
                    plt.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
                else:
                    plt.fill_between(x, scaling * mark, scaling * (mark + val), color=self.speech.data.colours[key])
            else:
                plt.fill_between(x, scaling * mark, scaling * (mark + val), label=key, color=self.speech.data.colours[key])
            mark += val
        plt.plot(x, scaling * mark, 'k')
        plt.legend(fontsize=12, loc='lower left', ncol=2)
        plt.xlim([0, np.max(x)])
        if set_ylim is None:
            plt.ylim([0, 1.1 * np.max(scaling * mark)])
        else:
            plt.ylim([0, set_ylim])
        plt.ylabel(unit, fontsize=14)
        plt.xlabel('Hour', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if title is not None:
            plt.set_title(title)
        if save_str is not None:
            plt.tight_layout()
            plt.savefig(save_str, bbox_inches='tight')
        plt.show()

