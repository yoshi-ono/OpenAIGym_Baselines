import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# 環境の生成
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env]) # (1)

# モデルの生成 (2)
model = PPO2('MlpPolicy', env, verbose=1)

# モデルの学習 (3)
model.learn(total_timesteps=128000)

# モデルのテスト
state = env.reset()
while True:
    # 環境の描画
    env.render()

    # モデルの推論 (4)
    action, _ = model.predict(state, deterministic=True)

    # 1ステップ実行
    state, rewards, done, info = env.step(action)

    # エピソード完了
    if done:
        break

# 環境のクローズ
env.close()