import gymnasium as gym
import numpy as np
from gymnasium import spaces
import economic_model

class CentralBankEnvironment(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        # action spaces (interest rate -> real numbers between -0.5 and 0.5)
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

        # observation spaces
        self.observation_space = spaces.Box(low=-2**63, high=2**63 - 2, shape=(10, 79), dtype=np.float32)

        # initial space
        self.space = 1

        # neuronal network which approximates the economy
        model_loader = economic_model.ModelLoader()
        self.economic_simulation = model_loader.model
    
    
