import gym

# 環境の生成
env = gym.make('CartPole-v1')

# 環境のリセット
state = env.reset()

# 1エピソードのループ
while True:
    # 環境の描画
    env.render()

    # ランダム行動の取得
    action = env.action_space.sample()

    # 1ステップの実行
    state, reward, done, info = env.step(action)
    print('reward:', reward)

    # エピソード完了
    if done:
        print('done')
        break

# 環境のクローズ
env.close()