import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# 環境の生成
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO2('MlpPolicy', env, verbose=1)

# モデルの学習
model.learn(total_timesteps=100000)

# モデルの保存 (1)
model.save('sample')