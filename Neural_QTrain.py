import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9  # discount factor
INITIAL_EPSILON = 0.1  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
EPSILON_DECAY_STEPS = 100  # decay period
LEARNING_RATE = 0.01 #learning reate

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

# TODO: Define Network Graph
def network(state_in, state_dim, action_dim, hidden_nodes = [50,50]):
	w1 = tf.get_variable("w1", shape=[state_dim, hidden_nodes[0]])
	b1 = tf.get_variable("b1", shape=[1, hidden_nodes[0]], initializer = tf.constant_initializer(0.0)) 

	w2 = tf.get_variable("w2", shape=[hidden_nodes[0], hidden_nodes[1]])
	b2 = tf.get_variable("b2", shape=[1, hidden_nodes[1]], initializer = tf.constant_initializer(0.0)) 

	w3 = tf.get_variable("w3", shape=[hidden_nodes[1], action_dim])
	b3 = tf.get_variable("b3", shape=[1, action_dim], initializer = tf.constant_initializer(0.0)) 

	l1_logits = tf.matmul(state_in, w1) + b1
	l1_out = tf.tanh(l1_logits)
	# option for tf.sigmoid
	
	l2_logits = tf.matmul(l1_logits, w2) + b2
	l2_out = tf.tanh(l2_logits)

	l3_logits = tf.matmul(l2_logits, w3) + b3
	l3_out = tf.tanh(l3_logits)
	
	return l3_out

# TODO: Network outputs
q_values = network(state_in, STATE_DIM, ACTION_DIM, [50,50])
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
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


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

	
	
        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        # target =
	# NEED TO DEFINE TARGET
        # Do one training step
        session.run([optimizer], feed_dict={
            target_in: [target],
            action_in: [action],
            state_in: [state]
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
