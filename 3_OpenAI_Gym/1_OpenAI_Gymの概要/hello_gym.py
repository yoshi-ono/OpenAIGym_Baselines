import gym

# 環境の生成
env = gym.make('CartPole-v1')

# ランダム行動による動作確認
env.reset()
while True:
    env.render()
    env.step(env.action_space.sample())