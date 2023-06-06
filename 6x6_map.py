
# Building enviroments

import gymnasium as gym
import numpy as np
import random

#1

m_66_1 =[
    "SFFFFF",
    "FFFFFF",
    "FFFFFF",
    "FFFFFF",
    "FFFFFF",
    "FFFFFG",
] 

env1 = gym.make("FrozenLake-v1",desc=m_66_1,map_name="6x6",is_slippery=False)

#2

m_66_2 =[
    "SFFFFF",
    "HHHHHF",
    "FFFFFF",
    "FHHHHH",
    "FFFFFF",
    "HHHHHG",
] 

env2 = gym.make("FrozenLake-v1",desc=m_66_2,map_name="6x6",is_slippery=False)

#3

m_66_3 =[
    "SFFFFF",
    "FHFHFH",
    "FFFFFF",
    "FHFHFH",
    "FFFFFF",
    "FHFHFG",
] 

env3 = gym.make("FrozenLake-v1",desc=m_66_3,map_name="6x6",is_slippery=False)


#4

m_66_4 =[
    "SFFFFF",
    "FHFHFH",
    "FFFFFF",
    "HHHHHF",
    "FFFFFF",
    "FHFHFG",
] 

env4 = gym.make("FrozenLake-v1",desc=m_66_4,map_name="6x6",is_slippery=False)

#5 

m_66_5 =[
    "SFFFFF",
    "FHFHFH",
    "FFFFFF",
    "FHHHHF",
    "FFFFFF",
    "FHFHFG",
] 

env5 = gym.make("FrozenLake-v1",desc=m_66_5,map_name="6x6",is_slippery=False)

#6

m_66_6 =[
    "SFFFFF",
    "HFFFFH",
    "HFHHFH",
    "HFHHFH",
    "HFFFFF",
    "HHHHHG",
] 

env6 = gym.make("FrozenLake-v1",desc=m_66_6,map_name="6x6",is_slippery=False)

m_66_7 =[
    "SFFHHH",
    "HHFHHH",
    "HHFHHH",
    "HHFHHH",
    "HHFFFF",
    "HHHHHG",
] 

env7 = gym.make("FrozenLake-v1",desc=m_66_7,map_name="6x6",is_slippery=False)


def qlearning(env,total_episodes):
    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))
    #total_episodes = 100000        # Total episodes
    learning_rate = 0.9           # Learning rate
    max_steps = 99                # Max steps per episode
    gamma = 0.95                  # Discounting rate

    # Exploration parameters
    epsilon = 1                # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 1/total_episodes             # Exponential decay rate for exploration prob
    # List of rewards
    rewards = []
    # 2 For life or until learning is stopped
    for episode in range(total_episodes):
        # Reset the environment
        state_reset = env.reset()
        state = state_reset[0]
        #print(state)
        step = 0
        done = False
        total_rewards = 0
        
        for step in range(max_steps):
            # 3. Choose an action a in the current world state (s)
            ## First we randomize a number
            exp_exp_tradeoff = random.uniform(0, 1)
            #print(exp_exp_tradeoff)
            ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])

            # Else doing a random choice --> exploration
            else:
                action = env.action_space.sample()
                #print(action)

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, trunc, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            #print(state,action,reward)
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            
            total_rewards += reward
            
            # Our new state is state
            state = new_state
            
            # If done (if we're dead) : finish episode
            if done == True: 
                break
            
        episode += 1
        # Reduce epsilon (because we need less and less exploration)
        
        #epsilon = epsilon*decay
        #epsilon = epsilon*0.999999
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
        rewards.append(total_rewards)

def Q_moving_check(env,qtable, number_of_episodes):

    env.reset()
    max_steps = 16
    for episode in range(number_of_episodes):
        state = env.reset()
        state = state[0]
        step = 0
        done = False
        #print("****************************************************")
        #print("EPISODE ", episode)
        #print(state)
        for step in range(max_steps):
            # env.render()
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(qtable[state,:])
            #print(action)
            new_state, reward, done, trunc, info = env.step(action)
            if done:
                if reward == 1:
                    return(reward)
                else:
                    reward = 0
                    return(reward)
                #print('epis length',step)
                #print('reward',reward)
                break
            if step == max_steps:
                reward = 0
                return(reward) 

            state = new_state

def q_performance(env,max_ep,data_collect,points_distance_accordingtodatacollect):
    ep = 0
    #success_list = np.array(list(range(0,data_collect)))
    success_list = np.array([])
    while ep<max_ep:
        rewards = 0 # total rewards per data collection section
        success_rate = 0 # success rate in each data sampling session 
        ep_it = int(data_collect/points_distance_accordingtodatacollect) # number of test points in each data collection session
        for s in range(ep_it): 
            ep = ep + points_distance_accordingtodatacollect # iteration through max episodes
            qtable = qlearning (env,ep)
            reward = Q_moving_check(env,qtable,1) or 0
            rewards +=  reward
        success_rate = rewards/ep_it
        success_list = np.append(success_list, success_rate) # you have to make a list which each element is average eward per data collect
        #return success_list
    return success_list

import matplotlib.pyplot as plt
import matplotlib as mpl


# 100000 50 data sets
# 2000 average for 50 points
# 40 points distance between each sample 


data_collect = 2000 # data collection rate
max_ep = 100000 # max episode for performance check
points_distance_accordingtodatacollect = 40 # how many points calculated in data collect session to reduce calculation costs 
# final average is based on this number

## performance check based on elapsed episodes (training)

success_list_1 = q_performance(env1,max_ep,data_collect,points_distance_accordingtodatacollect)
print('Env1 done')
success_list_2 = q_performance(env2,max_ep,data_collect,points_distance_accordingtodatacollect)
print('Env2 done')
success_list_3 = q_performance(env3,max_ep,data_collect,points_distance_accordingtodatacollect)
print('Env3 done')
success_list_4 = q_performance(env4,max_ep,data_collect,points_distance_accordingtodatacollect)
print('Env4 done')
success_list_5 = q_performance(env5,max_ep,data_collect,points_distance_accordingtodatacollect)
print('Env5 done')
success_list_6 = q_performance(env6,max_ep,data_collect,points_distance_accordingtodatacollect)
print('Env6 done')
ss_list_7 = q_performance(env7,max_ep,data_collect,points_distance_accordingtodatacollect)
print('Env7 done')

## ploting the data

data_points = max_ep/data_collect
data_points_array = np.array(list(range(0,int(data_points))))


fig, ax = plt.subplots()
plt.style.use('plot_style.txt')

plt.scatter(data_points_array,success_list_1,c='C1',edgecolor ="red",linewidths=0.1,marker='o',s=10, alpha=0.5,label='Env1')
plt.scatter(data_points_array,success_list_2,c='C2',edgecolor ="red",linewidths=0.1,marker='s',s=10, alpha=0.5,label='Env2')
plt.scatter(data_points_array,success_list_3,c='C3',edgecolor ="red",linewidths=0.1,marker='X',s=10, alpha=0.5,label='Env3')
plt.scatter(data_points_array,success_list_4,c='C4',edgecolor ="red",linewidths=0.1,marker='D',s=10, alpha=0.5,label='Env4')
plt.scatter(data_points_array,success_list_5,c='C5',edgecolor ="red",linewidths=0.1,marker='p',s=10, alpha=0.5,label='Env5')
plt.scatter(data_points_array,success_list_6,c='C6',edgecolor ="red",linewidths=0.1,marker='*',s=10, alpha=0.5,label='Env6')
plt.scatter(data_points_array,success_list_6,c='C6',edgecolor ="red",linewidths=0.1,marker='*',s=10, alpha=0.5,label='Env7')


#plt.title('Performance check for')
plt.xlabel('Elapsed episode pecentage')
plt.ylabel('Performance per data collect rate')

leg = ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Different maps')

plt.grid()
plt.show()