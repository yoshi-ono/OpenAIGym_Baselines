import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# 環境の生成
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

# モデルの読み込み (1)
model = PPO2.load('sample')

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