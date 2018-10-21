import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# HyperParameters
GAMMA = 0.9  # Discount factor
INITIAL_EPSILON = 1.0  # Starting value of epsilon
FINAL_EPSILON = 0.01  # Final value of epsilon
EPSILON_DECAY_STEPS = 100  # Decay period
LEARNING_RATE = 0.001  # Learning reate
PUNISH = -10  # Amount to negatively reward actions that cause termination

BATCH_SIZE = 32  # Size of each batch
MAX_SIZE_MEMORY = 10000  # Max number of experiences stored in memory

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])


# Network
def network(state_dim, action_dim, hidden_nodes):
    w1 = tf.get_variable(
        "w1", shape=[state_dim, hidden_nodes[0]])
    b1 = tf.get_variable(
        "b1", shape=[1, hidden_nodes[0]],
        initializer=tf.zeros_initializer)

    w2 = tf.get_variable(
        "w2", shape=[hidden_nodes[0], hidden_nodes[1]])
    b2 = tf.get_variable(
        "b2", shape=[1, hidden_nodes[1]],
        initializer=tf.zeros_initializer)

    w3 = tf.get_variable("w3", shape=[hidden_nodes[1], action_dim])
    b3 = tf.get_variable(
        "b3", shape=[1, action_dim], initializer=tf.zeros_initializer)

    l1_logits = tf.matmul(state_in, w1) + b1
    l1_out = tf.nn.relu(l1_logits)

    l2_logits = tf.matmul(l1_out, w2) + b2
    l2_out = tf.nn.relu(l2_logits)

    l3_logits = tf.matmul(l2_out, w3) + b3
    l3_out = l3_logits

    return l3_out


# Network outputs
q_values = network(STATE_DIM, ACTION_DIM, [30, 30])
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

# Memory to store experiences
memory = deque()

# Main learning loop
for episode in range(EPISODE):
    # Initialize task
    state = env.reset()
    # Update epsilon once per episode
    if epsilon > FINAL_EPSILON:
        epsilon -= epsilon / EPSILON_DECAY_STEPS
    else:
        epsilon = FINAL_EPSILON

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        # If memory is full, remove the oldest experience
        if len(memory) > MAX_SIZE_MEMORY:
            memory.popleft()

        # Place step into memory
        memory.append([state, action, next_state, reward, done])

        # Create a batch once we have enough experiences
        if len(memory) >= BATCH_SIZE:
            # Randomly sample a batch of experiences
            batches = random.sample(memory, BATCH_SIZE)

            # Create the states, actions and targets for each batch
            batch_state = []
            batch_action = []
            batch_target = []
            for s, a, ns, r, d in batches:
                batch_state.append(s)
                batch_action.append(a)
                batch_next_q = q_values.eval(feed_dict={
                    state_in: [ns]
                })
                # Calculate targets
                if d:
                    # Punish actions that cause termination
                    batch_target.append(PUNISH)
                else:
                    batch_target.append(
                        r + GAMMA * np.max(batch_next_q))

            # Do one training step
            session.run([optimizer], feed_dict={
                state_in: batch_state,
                target_in: batch_target,
                action_in: batch_action
            })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                ave_reward = total_reward / TEST
                print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                      'Average Reward:', ave_reward)

env.close()
