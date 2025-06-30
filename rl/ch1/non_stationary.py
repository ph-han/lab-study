import numpy as np
import matplotlib.pyplot as plt

class NonStatBandit:
    def __init__(self, arms=10): # arms = # of slotmachines
        self.arms = arms
        self.rates = np.random.rand(arms) # winning rate of each slotmachine

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms) # noise
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
class AlphaAgent:
    def __init__(self, epsilon, alpha, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilon = 0.1
    alpha = 0.8
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = NonStatBandit()
        agent = AlphaAgent(epsilon, alpha)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward

            rates.append(total_reward / (step + 1))
        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)

    print(total_reward)
    plt.ylabel('Total rewards')
    plt.xlabel('Steps')
    plt.plot(avg_rates)
    plt.show()