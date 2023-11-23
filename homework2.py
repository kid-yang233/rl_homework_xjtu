import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def p_possion(lemma):
    p_k = []
    for i in range(1,21):
        p_k.append(lemma**i*np.exp(-lemma)/math.factorial(i)) 
    
    return p_k


class state():
    def __init__(self,p1,p2,value = 0,best_action = 0):
        self.p1 = p1
        self.p2 = p2
        self.best_action = best_action
        self.value = value
        self.p = np.zeros((21,21),dtype=np.float32)

        r1 = p_possion(3)
        r2 = p_possion(4)#rent car

        self.reward = 0
        for i in range(len(r1)):
            self.reward += 10 * r1[i] * (i+1 if i+1<self.p1 else self.p1)
            self.reward += 10 * r2[i] * (i+1 if i+1<self.p2 else self.p2)
    
    def move_car(self,a):
        k = 0
        if self.p2 - a<0:
            a = self.p2
        if self.p1 + a<0:
            a = - self.p1

        consume = 2 * np.abs(a) + k

        p1 = a + self.p1
        p2 = self.p2 - a

        # if a>0:
        #     consume = 2*(a-1)
        # if self.p1>10:
        #     consume +=10
        # if self.p2>10:
        #     consume +=10


        if p1>20:
            p1 = 20
        if p2>20:
            p2 = 20       
        return p1,p2,consume
    
    
    def state_auto_update(self):
        r1 = p_possion(3)
        r2 = p_possion(4)#rent car

        b1 = p_possion(3)
        b2 = p_possion(2)#back car

        p1_v = np.zeros(21,dtype=np.float32)
        p2_v = np.zeros(21,dtype=np.float32)

        for i in range(len(r1)):
            for j in range(len(b1)):
                t = self.p1 - (i+1) + (j+1)
                u = self.p2 - (i+1) + (j+1)
                if t<0:
                    t = 0
                elif t>20:
                    t = 20
                p1_v [t] += r1[i] * b1[j]

                if u<0:
                    u = 0
                elif u>20:
                    u = 20
                p2_v [u] += r2[i] * b2[j]

        p1_v = p1_v.reshape(21,1)
        p2_v = p2_v.reshape(1,21)

        self.p = np.dot(p1_v,p2_v)

        return self.p    
    
    def _value(self,x,y):
        reward = 0
        if x>self.p1:
            reward += 10*x - self.p1
        if y>self.p2:
            reward += 10*y - self.p1
        return reward


def state_value(state,state_all):
    value = np.zeros(11) 
    de = 0
    for x in range(21):
        for y in range(21):
            for i in range(11):
                if x + action[i] <0 or y - action[i]<0:
                    continue    
                u,v,consume =  state_all[x][y].move_car(action[i])
                value[i] += state.p[x][y] * (state_all[u][v].reward- consume + lemma * state_all[u][v].value )    
    g = state.value
    state.value  = np.amax(value)

    state.best_action = action[np.argmax(value)]
    de = state.value - g
    
    return de



lemma = 0.9
action = range(-5,6)

def main():
    state_all  = []
    for i in range(21):
        t = []
        for j in range(21):
            k = state(i,j)
            k.value = k.reward
            p = k.state_auto_update()
            t.append(k)
        state_all.append(t)
    
    epochs = 30
    best_action_set = np.zeros((epochs,21,21))
    delta = []
    for epoch in range(epochs):
        e = 0
        for i in range(21):
            for j in range(21):
                e += state_value(state=state_all[i][j],state_all=state_all)
                best_action_set[epoch,i,j] = state_all[i][j].best_action
        
        delta.append(e)


    return best_action_set,delta

if __name__ == "__main__":
    best_action_set,delta = main()
    color = ['c', 'b', 'g', 'r']
    marker = ['.','>','o','s','*']

    epochs = best_action_set.shape[0]
    let = [0,1,2,3,epochs-1]
    lang = range(-5,6)

    fig,ax = plt.subplots()
    ax.plot(range(epochs),delta)
    ax.set_xlabel('epoch')
    ax.set_ylabel('delta')
    plt.savefig('homework2/task1/delta.png')
    ax.clear()
    plt.close(fig)

    for epoch in let:
        x = []
        y = []
        for t in range(-5,6):
            a = []
            b = []
            x.append(a)
            y.append(b)
        for i in range(21):
            for j in range(21):
                x[int(best_action_set[epoch][i][j]+5)].append(i)
                y[int(best_action_set[epoch][i][j]+5)].append(j)
        fig,ax = plt.subplots()
        u = []
        for i in range(11):
            u.append(ax.scatter(x[i],y[i],c=color[i%4],marker=marker[i % 5]))
        plt.legend(handles = u,labels = action)
        ax.set_title('epoch = {}'.format(epoch+1),fontsize=20)    
        ax.set_xlabel('postion 1 ')
        ax.set_ylabel('postion 2 ')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.savefig('homework2/task1/epoch{}.png'.format(epoch+1))
        ax.clear()
        plt.close(fig)
