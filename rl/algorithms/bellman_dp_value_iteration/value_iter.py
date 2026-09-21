from grid_world import GridWorldEnv
from agent import Agent
from config import UP, DOWN, LEFT, RIGHT

def value_iter_onestep(V: dict, env: GridWorldEnv, gamma: float) -> None:
    for state in env.states():
        if state in env.finish_states():
            V[state] = 0
            continue

        action_values = []
        for action in env.actions:
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)



def value_iter(agent: Agent, env: GridWorldEnv, threshold=0.001):
    while True:
        old_V = agent.V.copy()
        value_iter_onestep(agent.V, env, agent.gamma)

        delta = 0
        for state in V.keys():
            t = abs(agent.V[state] - old_V[state])
            delta = max(delta, t)

        if delta < threshold:
            break
