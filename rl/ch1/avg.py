import numpy as np

np.random.seed(0) # seed fixed
rewards = []

# basic
for n in range(1, 11): # 10 plays
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(f"step {n} - {Q}") # Q value of each play

print("-" * 40)

# more efficiant
Q = 0
for n in range(1, 11):
    reward = np.random.rand()
    Q = Q + (reward - Q) / n
    print(f"step {n} - {Q}")
