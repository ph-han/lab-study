from grid_world import GridWorldEnv
from agent import Agent

def eval_one_step(pi: dict, V: dict, env: GridWorldEnv, gamma: float) -> dict:
    for state in env.states():
        if state in env.finish_states:
            V[state] = 0
            continue

        current_state_action_probs = pi[state]
        V_s = 0
        for action, action_prob in current_state_action_probs.items():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            V_s += action_prob * (reward + gamma * V[next_state])

        V[state] = V_s

    return V


def policy_eval(agent: Agent, env: GridWorldEnv, threshold: float):
    while True:
        old_V = agent.V.copy()
        eval_one_step(agent.pi, agent.V, env, agent.gamma)

        delta = 0
        for state, value in agent.V.items():
            t = abs(agent.V[state] - old_V[state])
            delta = max(delta, t)

        if delta <= threshold:
            break
