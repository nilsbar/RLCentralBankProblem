# RLCentralBankSimulation

The goal of this project is to create a Reinforcement Learning problem where the enviroment models the economy and the agent is the central bank. 

Model:

state-space: economical data of the past 10 years

action-space: interest rate

reward: |2 - inflation rate|

The transition function is a neural network which computes the economical data for the next year given the state, adds the action to it, deletes the first data and add the current data as the next data to the state.

Data sources:

CPI (-> transform them in to the inflation rate): 

Data for economy modelling: https://www.cbo.gov/data/budget-economic-data

Data for interest rates: https://fred.stlouisfed.org/series/FEDFUNDS

Problems:

- time series model for the economy needs to be adjusted



