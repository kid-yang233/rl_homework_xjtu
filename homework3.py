import numpy as np
import math
import matplotlib.pyplot as plt

alpha = 0.2
e = 0.8
per = 0.1

def choose_max(state_action,state_action_value):
    q = []

    for i in range(len(state_action)):
        q.append(state_action_value[i])
        if state_action[i] == 0 :
            q[i] = -999999999

    action = np.argmax(q)        

    return action

def move(state,action):
    new_state = []
    new_state.append(state[0])
    new_state.append(state[1]) 
    if action == 1: #1 right
        new_state[1]+=1
    elif action ==2: #2 left
        new_state[1]-=1
    elif action ==3: #3 up
        new_state[0]+=1
    elif action ==0: #0 down
        new_state[0]-=1
    return new_state

def new(state_action_value,state_action):
    q = []
    for i in range(4):
        if state_action[i] != 0:
            q.append(state_action_value[i])
    if q == []:
        return -99
    q = np.array(q)
    return np.average(q)


class cliff():
    def __init__(self):
        self.reward = np.ones((4,12),dtype=np.float16) * (-1)
        self.start = [0,0]
        self.end = [0,11]
        for i in range(1,12):
            self.reward[0,i] = -100
        self.reward[0,-1] = 0
        self.action_choose = np.ones((4,12),dtype=int) 
        self.state_action = np.ones((4,12,4),dtype=int) 
        self.state_action_value = np.zeros((4,12,4),dtype=np.float32) 
        self.state_action[0,:,0] = np.zeros_like(self.state_action[0,:,0])
        self.state_action[3,:,3] = np.zeros_like(self.state_action[3,:,3])
        self.state_action[:,0,2] = np.zeros_like(self.state_action[:,0,2])
        self.state_action[:,11,1] = np.zeros_like(self.state_action[:,11,1])


    
    def greedy(self):
        pie = []
        state = [0,0]
        state_value = np.ones((4,12))
        for i in range(4):
            for j in range(12):
                state_value[i,j] = new(self.state_action_value[i,j],self.state_action[i,j]) #结果不同
                # state_value[i,j] = np.max(self.state_action_value[i,j])  #结果同样
        state_value[0,11] = 100
        while state!= [0,11]:
            if state[0] ==0 and (state[1]<=10 and state[1]>=1):
                next_state = [0,0]
                action = -1
            else:
                next_state,action = choose(state,state_value,self.state_action)
                state_value[next_state[0],next_state[1]] += -1
                self.action_choose[state[0],state[1]] = action
            if next_state[0] ==0 and (next_state[1]<=10 and next_state[1]>=1):
                state_value[next_state[0],next_state[1]] += -99
            state = next_state
            pie.append([next_state,action])
        return pie
            


def choose(state,state_value,state_action):
    value_action = []
    value = []
    u = state 
    for i in range(4):
        if state_action[state[0],state[1],i]!=0:
            next_state = move(u,i)
            value_action.append([next_state,i])
            value.append(state_value[next_state[0],next_state[1]])
    value = np.array(value)
    k = np.argmax(value)
    next_state,action = value_action[k]
    return next_state,action


def sarsa(sel,expiose):
    reward_all = []
    for exp in range(expiose):
        reward = 0
        state = [0,0]
        pie = sel.greedy()                 
        for next_state,action in pie:
            r = -1
            next_state_action = sel.action_choose[next_state[0],next_state[1]]
            # next_state_action = choose_max(self.state_action[next_state_action[0],next_state_action[1]],self.state_action_value[next_state_action[0],next_state_action[1]])
            if state[0] ==0 and (state[1]<=10 and state[1]>=1):
                sel.state_action_value[state[0],state[1],:] = (sel.state_action_value[state[0],state[1],:] + alpha *(-100*np.ones_like(sel.state_action_value[state[0],state[1],:])))
                state = next_state 
                continue 
            if next_state[0] ==0 and (next_state[1]<=10 and next_state[1]>=1):
                r = -100
            sel.state_action_value[state[0],state[1],action] = (
                sel.state_action_value[state[0],state[1],action] + alpha *(
                r + e * sel.state_action_value[next_state[0],next_state[1],next_state_action] - sel.state_action_value[state[0],state[1],action]
                    )
                )
            reward += 0 if state == [0,11] else r
            state = next_state         
        reward_all.append(reward)

    return reward_all
    

def q_learning(sel,expiose):
    reward_all = []
    for exp in range(expiose):
        reward = 0
        state = [0,0]
        pie = sel.greedy()                 
        for next_state,action in pie:
            r = -1 
            # next_state_action = sel.action_choose[next_state[0],next_state[1]]
            next_state_action = choose_max(sel.state_action[next_state[0],next_state[1]],sel.state_action_value[next_state[0],next_state[1]])
            if state[0] ==0 and (state[1]<=10 and state[1]>=1):
                sel.state_action_value[state[0],state[1],:] = (sel.state_action_value[state[0],state[1],:] + alpha *(-100*np.ones_like(sel.state_action_value[state[0],state[1],:])))
                state = next_state
                continue 
            if next_state[0] ==0 and (next_state[1]<=10 and next_state[1]>=1):
                r = -100
            sel.state_action_value[state[0],state[1],action] = (
                sel.state_action_value[state[0],state[1],action] + alpha *(
                r + e * sel.state_action_value[next_state[0],next_state[1],next_state_action] - sel.state_action_value[state[0],state[1],action]
                    )
                )
            reward += 0 if state == [0,11] else r
            state = next_state         
        reward_all.append(reward)

    return reward_all
    
      

        
a1 = cliff()
a2 = cliff()
q1 = sarsa(a1,30)
q2 = q_learning(a2,30)
fig,ax = plt.subplots()
print(q1)
print(q2)
u1, = ax.plot(range(1,30),q1[1:],color='b',marker='o')
u2, = ax.plot(range(1,30),q2[1:],color = 'r',marker='x')
ax.set_title('reward')
ax.legend([u1,u2],labels=['sarsa','q_learning'])
plt.savefig('homework/RL/homework3/new/result')