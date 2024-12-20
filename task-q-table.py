import numpy as np
import gym

# Создаем окружение
env = gym.make('CartPole-v1')  # Используем CartPole-v1
np.bool8 = np.bool_
# Параметры Q-learning
num_episodes = 300
max_steps = 200
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01

num_buckets = (6, 12, 6, 12)
q_table = np.random.uniform(low=-1, high=1, size=(num_buckets + (env.action_space.n,)))


def discretize_state(state):
    # state уже является массивом
    position, velocity, angle, angular_velocity = state

    state_indices = []
    state_indices.append(np.digitize(position, np.linspace(-2.4, 2.4, num_buckets[0] - 1)))
    state_indices.append(np.digitize(velocity, np.linspace(-3.0, 3.0, num_buckets[1] - 1)))
    state_indices.append(np.digitize(angle, np.linspace(-0.5, 0.5, num_buckets[2] - 1)))
    state_indices.append(np.digitize(angular_velocity, np.linspace(-2.0, 2.0, num_buckets[3] - 1)))
    return tuple(state_indices)


# В основном цикле
for episode in range(num_episodes):
    state = env.reset()
    print("Initial state:", state)
    state = discretize_state(state[0])  # Передаем только первое значение, которое содержит массив состояния

    for step in range(max_steps):
        if np.random.rand() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_state)  # Передаем next_state напрямую

        q_table[state][action] += learning_rate * (
                reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action])

        state = next_state

        if terminated or truncated:
            break

    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

# Проверка успешных эпизодов
success_count = 0
for episode in range(200):
    state = env.reset()
    state = discretize_state(state[0])  # Передаем только первое значение

    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_state)  # Передаем next_state напрямую

        if terminated or truncated:
            success_count += 1
            break

print(f"Успешных эпизодов: {success_count} из 200")

env.close()
