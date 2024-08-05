import gymnasium as gym
import numpy as np
from gymnasium import spaces
import economic_model
import torch

class CentralBankEnvironment(gym.Env):
    
    def __init__(self) -> None:
        super().__init__()

        # action spaces (interest rate -> real numbers between -0.5 and 0.5)
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

        # time horizon of the lookback
        self.lookback = 10

        # number of parameters of the economy (interest rate included)
        economical_parameters = 78 

        # observation spaces
        self.observation_space = spaces.Box(low=-2**63, high=2**63 - 2, shape=(self.lookback, economical_parameters), dtype=np.float32)

        # initial space 
        self.state = np.loadtxt('initial_state.txt', delimiter=',').astype(np.float32)

        # neuronal network which approximates the economy
        model_loader = economic_model.ModelLoader()
        self.economic_model = model_loader.model

        # maximum steps
        self.step_counter = 0
        self.max_steps = 20
    
    def step(self, action):
        
        # economic_simulation
        input_tensor = torch.from_numpy(self.state)
        input_tensor = input_tensor.to(torch.float32)
        with torch.no_grad():  # Keine Gradientenberechnung erforderlich
            output_tensor = self.economic_model(input_tensor)[-1,:]
        economic_output = output_tensor.cpu().detach().numpy()

        # inflation rate
        inflation_rate = economic_output[-1]

        # new economic result is the output of the model + action
        new_economic = np.append(economic_output, action)
        new_economic = new_economic[np.newaxis, :] # transform the 1-dim array to a 2-dim with one row

        # new state is the first row dropped and new_economic added below
        self.state = self.state[1:]
        self.state = np.append(self.state, new_economic, axis=0)

        # reward |2% - inflation rate|
        reward = abs(2 - inflation_rate)

        terminated = False
        
        if self.step_counter <= self.max_steps:
            truncated = False
        else: 
            truncated = True

        info = {} 

        return self.state, reward, terminated, truncated, info
    
    def reset(self, seed = None):

        self.space = np.loadtxt('initial_state.txt', delimiter=',').astype(np.float32)

        return self.space, {}
