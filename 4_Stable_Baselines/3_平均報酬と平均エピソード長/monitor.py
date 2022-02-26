import gym
import os
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.bench import Monitor

# ログフォルダの生成 (1)
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# 環境の生成
env = gym.make('CartPole-v1')
env = Monitor(env, log_dir, allow_early_resets=True) # (2)
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO2('MlpPolicy', env, verbose=1)

# モデルの学習
model.learn(total_timesteps=100000)

# モデルのテスト
state = env.reset()
for i in range(200):
    # 環境の描画
    env.render()

    # モデルの推論
    action, _ = model.predict(state)

    # 1ステップ実行
    state, rewards, done, info = env.step(action)

    # エピソード完了
    if done:
        break

# 環境のクローズ
env.close()