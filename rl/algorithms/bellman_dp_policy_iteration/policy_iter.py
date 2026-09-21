from grid_world import GridWorldEnv
from agent import Agent
from config import UP, DOWN, LEFT, RIGHT
from policy_eval import policy_eval

def argmax(d: dict):
    if not d:
        return None
    return max(d, key=d.get)

def greedy_policy(V: dict, env: GridWorldEnv, gamma: float) -> dict:
    pi = {}
    for state in env.states():
        actions_value = {}
        for action in env.actions:
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            actions_value[action] = reward + gamma * V[next_state]

        greedy_action = argmax(actions_value)
        action_probs = {UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0}
        action_probs[greedy_action] = 1
        pi[state] = action_probs

    return pi
        

def policy_iter(agent: Agent, env: GridWorldEnv, threshold: float):
    while True:
        policy_eval(agent, env, threshold)
        new_pi = greedy_policy(agent.V, env, agent.gamma)

        if new_pi == agent.pi:
            break

        agent.pi = new_pi
        
