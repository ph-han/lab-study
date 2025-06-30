import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent
from non_stationary import NonStatBandit, AlphaAgent

runs = 200
steps = 1000
epsilon = 0.1
alpha = 0.8
all_rates = np.zeros((runs, steps))
all_rates2 = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    nonStatBandit = NonStatBandit()
    alphaAgent = AlphaAgent(epsilon, alpha)
    total_reward = 0
    total_reward2 = 0
    rates = []
    rates2 = []

    for step in range(steps):
        action = agent.get_action()
        action2 = alphaAgent.get_action()
        reward = bandit.play(action)
        reward2 = nonStatBandit.play(action2)
        agent.update(action, reward)
        alphaAgent.update(action2, reward2)
        total_reward += reward
        total_reward2 += reward2

        rates.append(total_reward / (step + 1))
        rates2.append(total_reward2 / (step + 1))
    all_rates[run] = rates
    all_rates2[run] = rates2

avg_rates = np.average(all_rates, axis=0)
avg_rates2 = np.average(all_rates2, axis=0)

print(total_reward)
plt.ylabel('Total rewards')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.plot(avg_rates2)
plt.show()