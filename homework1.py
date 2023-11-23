import numpy as np
import matplotlib.pyplot as plt

class machine():
    def __init__(self,k=10,aver=0,sigma=0.01,n=10000,e=0.1,step=0.1,step_open = False):
        self.k = k

        self.aver = aver
        self.sigma = sigma
        self.n = n

        self.r = np.zeros(n)
        self.q = np.zeros(k)
        self.q_star = np.ones(k)
        self.N = np.zeros(k)
        self.best_action = np.zeros(n)

        self.step = step
        for i in range(n):
            self.q_star = self.q_star + np.random.normal(0,0.01,k)

            a = np.random.rand()
            best_choose = np.argmax(self.q_star)

            if a > e :
                choose = np.argmax(self.q)
            else:
                choose = np.random.randint(0,k)
            if i==0:
                self.r[i] = self.q_star[choose]
            else:
                self.r[i] = self.r[i-1] + (self.q_star[choose] - self.r[i-1]) / (i+1)

            if step_open == True:
                self.N[choose]+=1
                self.q[choose] = self.q[choose] + step * (self.q_star[choose] - self.q[choose])
            else:
                self.N[choose]+=1
                self.q[choose] = self.q[choose] + (self.q_star[choose] - self.q[choose])/self.N[choose]
            
            self.best_action[i] = 1 if choose == best_choose else 0



t = 10000
e = 2000

a1_aver_r = np.zeros(t)
a2_aver_r = np.zeros(t)
a1_action = np.zeros(t)
a2_action = np.zeros(t)

for i in range(e):
    a1 = machine(n=t)
    a2 = machine(n=t,step_open=True)
    a1_aver_r = a1_aver_r + (a1.r - a1_aver_r) /(i+1)
    a2_aver_r = a2_aver_r + (a2.r - a2_aver_r) /(i+1)
    a1_action = a1_action + (a1.best_action - a1_action)/(i+1)
    a2_action = a2_action + (a2.best_action - a2_action)/(i+1)


time = np.arange(0,t)



plt.plot(time,a1_aver_r,color='green',linewidth=2, markersize=12,label="sampling averge")
plt.plot(time,a2_aver_r,color='blue',linewidth=2, markersize=12,label="exponential constant")
plt.legend()
plt.ylabel(u'averge_reward') 
plt.xlabel(u'time')
plt.savefig('homework1.png')
plt.clf()

plt.plot(time,a1_action,color='green',linewidth=2, markersize=12,label="sampling averge")
plt.plot(time,a2_action,color='blue',linewidth=2, markersize=12,label="exponential constant")
plt.legend()
plt.ylabel(u'best_action percent') 
plt.xlabel(u'time')
plt.savefig('homework1a.png')
