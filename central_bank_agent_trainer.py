import gymnasium as gym
import central_bank_environment
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

env = central_bank_environment.CentralBankEnvironment()
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50)