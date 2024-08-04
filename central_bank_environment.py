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

        # observation spaces
        self.observation_space = spaces.Box(low=-2**63, high=2**63 - 2, shape=(10, 79), dtype=np.float32)

        # initial space
        self.space = np.loadtxt('initial_state.txt', delimiter=',')

        # neuronal network which approximates the economy
        model_loader = economic_model.ModelLoader()
        self.economic_model = model_loader.model
    
    def step(self, action):
        
        # economic_simulation
        input_tensor = torch.from_numpy(self.state)
        torch.from_numpy(input_tensor)
        input_tensor = input_tensor.to(torch.float32)
        with torch.no_grad():  # Keine Gradientenberechnung erforderlich
            output_tensor = self.economic_model(input_tensor)[-1,:]
        economic_output = output_tensor.cpu().detach().numpy()

        # new economic result is the output of the model + action
        new_economic = economic_output 

        # new state is the first row dropped and new_economic added below

        # observation model simulation + action added
        observation = None

        # reward |2% - inflation rate|
        reward = abs(2 - 0)

        terminated = None

        truncated = None

        info = None 

        return observation, reward, terminated, truncated, info
    
    def reset(self):

        self.space = np.loadtxt('initial_state.txt', delimiter=',')
