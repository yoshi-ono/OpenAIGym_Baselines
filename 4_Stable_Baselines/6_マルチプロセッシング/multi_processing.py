import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds

# 定数
ENV_ID = 'CartPole-v1' # 環境ID
NUM_ENV = 4 # 環境数

# 環境を生成する関数 (1)
def make_env(env_id, rank, seed=0):
   def _init():
       env = gym.make(env_id)
       env.seed(seed + rank)
       return env
   set_global_seeds(seed)
   return _init

# メイン関数の定義 (4)
def main():
    # 訓練環境の生成 (2)
    train_env = SubprocVecEnv([make_env(ENV_ID, i) for i in range(NUM_ENV)])

    # エージェントの生成
    model = PPO2('MlpPolicy', train_env, verbose=1)

    # モデルの学習
    model.learn(total_timesteps=10000)

    # テスト環境の生成 (3)
    test_env = DummyVecEnv([lambda: gym.make(ENV_ID)])

    # モデルのテスト
    state = test_env.reset()
    for i in range(200):
        # 環境の描画
        test_env.render()

        # モデルの推論
        action, _ = model.predict(state)

        # 1ステップ実行
        state, rewards, done, info = test_env.step(action)

        # エピソード完了
        if done:
            break

    # 環境のクローズ
    env.close()

# メインの実装 (4)
if __name__ == "__main__":
    main()