import gym
import loggol

loggol.Init()
logger = loggol.GetLogger()

# 環境の生成
env = gym.make('CartPole-v1')

# 環境のリセット
state = env.reset()

# 1エピソードのループ
while True:
    # 環境の描画
    env.render()

    # ランダム行動の取得
    #action = env.action_space.sample()
    action = 0
    logger.debug('action: %s', action)

    # 1ステップの実行
    state, reward, done, info = env.step(action)
    logger.debug('state: %s', state)
    logger.debug('reward: %s', reward)

    # エピソード完了
    if done:
        logger.debug('done')
        break

# 環境のクローズ
env.close()