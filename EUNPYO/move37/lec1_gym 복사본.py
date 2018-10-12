# gym 놀이
# 1) 만능 RL ALGORYTHM 만들기
# 2) 각 게임 environment에서 나오는 observation space 와 action space 의 상관관계 찾

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# CartPole 환경 만들어서 env에 저장
# max_episode_step =500

gym.envs.register(
    id='Cartpole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=1000,
    reward_threshold=-110.0,
)
env = gym.make('Cartpole-v0')


# 에피소드를 돌리는 함수 정의

def run_episode(env, parameters):
    observation = env.reset() #observation 값 초기화
    totalreward = 0 # totalreward 값 초기화
    for _ in range(1000): # max_episode_step 을 1000으로 set했으니
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

run_episode(env,[0.1,0.1,1,1])

# mean of 100 consecutive trial
def hund_episode_mean(env,parameters) :
    sum_reward=0
    for i in range(100):
        reward_ith=run_episode(env,parameters)
        sum_reward=sum_reward+reward_ith
    return sum_reward/100

hund_episode_mean(env,[0.1,0.1,1,1])
# (원래는 이렇게 잘 나오면 안 되는것응;;) 띠용;; 왤캐 점수가 잘 나오는것인가;;
# cartpole 설명을 읽어보면(200점 만점 default) 100번 try의 평균 득점이 195점 이상시 문제를 풀었다고 간주
# 1000점 만점인 내 기준에서는 평균이 975점 이상 나와야 문제를 푼 것;;
# 하여튼;; 거진 풀었네;; 카트폴은쉬운 문제였던거시다;;


# algorithm -> output 으로 solving_time 값 나오면 됨.

# Algorithm 1 - random search
def rand_search(): # assigning random parameter
    bestparams = None
    bestreward = 0
    for _ in range(1,202):
        parameters = np.random.rand(4)*2-1
        reward = hund_episode_mean(env,parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            # considered solved if the agent lasts 980 timesteps
            if reward >= 980:
                break
    if _ == 201: _ = 999
    print(bestparams)
    print(_)
    return _

# Algorithm 2 - hill climbing
def hill_climbing(noise_scale): # assigning random parameter
    bestparam  = None
    bestreward = 0
    parameters = np.random.rand( 4 ) * 2 - 1
    for _ in range(1,202):
        newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scale
        reward = hund_episode_mean(env, newparams)
        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward > 980:
                break
    if _ == 201: _ = 999
    print(parameters)
    print(_)
    return _


# 기록 - algorithm 1 - random search
# record the solve_time
result_rand_search=[]
for i in range(100) :
    temp=rand_search()
    result_rand_search.append(temp)

# histogram
plt.hist(result_rand_search,bins=20)
plt.show()

# 기록 - algorithm 2 - hill climbing
# record the solve_time
result_hill=[]
for i in range(100) :
    temp=hill_climbing(0.1)
    result_hill.append(temp)

# histogram
plt.hist(result_hill,bins=20)
plt.show()


