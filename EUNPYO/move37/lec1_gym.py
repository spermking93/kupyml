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
    max_episode_steps=200,
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


for _ in range(100) :
    state1 = env.reset()
    reward1 = env.step()
    action1 = np.random.random_integers(0,1,1)[0]




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

##################################################################################################
##################################################################################################

# Algorithm 3 - Policy Gradient

#Instead of only having one linear combination,
# we have two: one for each possible action.
# Passing these two values through a softmax function gives the probabilities of
# taking the respective actions, given a set of observations.
# This also generalizes to multiple actions, unlike the threshold we were using before.

# So we know how to update our policy to prefer certain actions.
class policy_gradient :
    params = tf.get_variable("policy_parameters",[4,2])
    state = tf.placeholder("float",[None,4])
    actions = tf.placeholder("float",[None,2])
    linear = tf.matmul(state,params)
    probabilities = tf.nn.softmax(linear)
    good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
    # maximize the log probability
    log_probabilities = tf.log(good_probabilities)
    loss = -tf.reduce_sum(log_probabilities)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# We a define some value for each state,
# that contains the average return starting from that state.
# In this example, we'll use a 1 hidden layer neural network.


class value_gradient():
    # sess.run(calculated) to calculate value of state
    state = tf.placeholder("float",[None,4])
    w1 = tf.get_variable("w1",[4,10])
    b1 = tf.get_variable("b1",[10])
    h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
    w2 = tf.get_variable("w2",[10,1])
    b2 = tf.get_variable("b2",[1])
    calculated = tf.matmul(h1,w2) + b2

    # sess.run(optimizer) to update the value of a state
    newvals = tf.placeholder("float",[None,1])
    diffs = calculated - newvals
    loss = tf.nn.l2_loss(diffs)
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

# tensorflow operations to compute probabilties for each action, given a state

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    pl_state = policy_gradient.state
    pl_probabilities = policy_gradient.probabilities
    observation = env.reset()
    actions = []
    transitions = []
    states = []
    totalreward=0
    for _ in range(200):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_probabilities,feed_dict={pl_state: obs_vector})
        rand_0_1 = tf.random_uniform( shape=[1], minval=0, maxval=1, dtype=tf.float32 )
        action = tf.cast( tf.less( rand_0_1, probs[0][1]),dtype=tf.int32 )
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation # 기존에 env.reset()으로 받아온 observation은 말하자면 S0(액션 이전)
        observation, reward, done, info = env.step(action) # 여기서 나온 observation은 액션 이후
        transitions.append((old_observation, action, reward))
        totalreward += reward
        if done:
            break