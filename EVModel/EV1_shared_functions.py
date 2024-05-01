import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
import os


def preprocess(df):
    
    df['Energy (kWh)'] = np.clip(np.abs(df['Energy (kWh)']), 0, 100)
    df['start_seconds'] = np.mod(df['start_seconds'], 24*3600)
    df['Session Time (secs)'] = np.clip(df['Session Time (secs)'], 0, 48*3600)
    df['start_15min'] = (df['start_seconds'] / 60 / 15).astype(int)
    df['Session Time (15min)'] = np.ceil(df['Session Time (secs)'] / 60 / 15).astype(int)
    df['Session Time (15min)'] = np.maximum(df['Session Time (15min)'], np.ceil(df_home['Energy (kWh)'] / 6.6 * 4)) # kwh / kw = h, h * (steps/hour) = steps
    
    return df

def end_times_and_load(start_times, energies, durations, rate):

    time_steps_per_hour = 4#60
    num_time_steps = 96#1440
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
        
    true_end_times = np.zeros(np.shape(start_times)).astype(int)
    inds1 = np.where((start_times + durations) > num_time_steps)[0]
    inds2 = np.delete(np.arange(0, np.shape(end_times)[0]), inds1)
    true_end_times[inds1] = (np.minimum(start_times[inds1].astype(int)+durations[inds1]-num_time_steps, num_time_steps)).astype(int)
    true_end_times[inds2] = (start_times[inds2] + durations[inds2]).astype(int)

    return true_end_times, load


class LoadModel(object):
    def __init__(self, num_sessions=1, charge_rate=6.6, solver='MOSEK'):
        """This method initializes many of the input and output variables used."""

        self.uncontrolled_total_load = np.zeros((1, ))  # The aggregate uncontrolled load profile
        self.controlled_total_load = np.zeros((1, ))
        self.num_sessions = num_sessions  # The number of sessions / number of cars in the parking lot that day
        self.arrival_inds = np.zeros((num_sessions,))  # The arrival time of each session, expressed as an index between 0 and 95
        self.departure_inds = np.zeros((num_sessions,))  # The index of the departure time for each vehicle
        self.energies = np.zeros((num_sessions,))  # The energy delivered in each uncontrolled session
        self.charge_rate = charge_rate  # The charge rate allowed. The default is level 2, 6.6 kW
        if solver == 'ECOS':
            self.solver = cvx.ECOS
        elif solver == 'MOSEK':
            self.solver = cvx.MOSEK
        else:
            self.solver = cvx.MOSEK
        self.time_steps_per_hour = 4#60
        self.num_time_steps = 96#1440
        
    def input_data(self, uncontrolled_load, start_inds, end_inds, energies):
        """Here the data about the uncontrolled load is provided and the data is preprocessed."""
        
        self.uncontrolled_total_load = uncontrolled_load
        self.arrival_inds = start_inds
        self.departure_inds = end_inds
        inds1 = np.where(self.departure_inds >= self.arrival_inds)[0]
        inds2 = np.where(self.departure_inds < self.arrival_inds)[0]
        session_length = np.zeros(np.shape(self.departure_inds))
        session_length[inds1] = self.departure_inds[inds1] - self.arrival_inds[inds1]
        session_length[inds2] = self.num_time_steps - self.arrival_inds[inds2] + self.departure_inds[inds2]
        energies_true = np.minimum(energies, (1/self.time_steps_per_hour)*self.charge_rate*(session_length))
        self.energies = energies_true
        
    def aef_controlled_load(self, energy_prices, verbose=False, reg=0):
        """Minimize for average emissions intensity in the grid, given here with proxy 'energy_prices'."""

        schedule = cvx.Variable((self.num_time_steps, self.num_sessions))
        obj = cvx.matmul(cvx.sum(schedule, axis=1), energy_prices.reshape((np.shape(energy_prices)[0], 1)))
        
        constraints = [schedule >= 0, schedule <= 6.6]
        for i in range(self.num_sessions):
            if self.departure_inds[i] >= self.arrival_inds[i]:
                if self.arrival_inds[i] > 0:
                    constraints += [schedule[np.arange(0, int(self.arrival_inds[i])), i] <= 0]
                if self.departure_inds[i] < self.num_time_steps:
                    constraints += [schedule[np.arange(int(self.departure_inds[i]), self.num_time_steps), i] <= 0]
            else:
                constraints += [schedule[np.arange(int(self.departure_inds[i]), int(self.arrival_inds[i])), i] <= 0]
        constraints += [(1/self.time_steps_per_hour) * cvx.sum(schedule, axis=0) == self.energies]
        prob = cvx.Problem(cvx.Minimize(obj + reg*cvx.sum_squares(cvx.sum(schedule, axis=1)[:-1] - cvx.sum(schedule, axis=1)[1:])), constraints)
        result = prob.solve(solver=self.solver)

        self.solar_controlled_power = schedule.value
         
        