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

# TODO: HyperParameters
GAMMA = 0.9  # discount factor
INITIAL_EPSILON = 0.6  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
EPSILON_DECAY_STEPS = 100  # decay period
LEARNING_RATE = 0.005 #learning reate

BATCH_SIZE = 16
MAX_SIZE_BUFFER = 10000
buffer = []

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
def network(state_dim, action_dim, hidden_nodes = [50,50]):
	w1 = tf.get_variable("w1", shape=[state_dim, hidden_nodes[0]])
	b1 = tf.get_variable("b1", shape=[1, hidden_nodes[0]], initializer = tf.constant_initializer(0.0))

	w2 = tf.get_variable("w2", shape=[hidden_nodes[0], hidden_nodes[1]])
	b2 = tf.get_variable("b2", shape=[1, hidden_nodes[1]], initializer = tf.constant_initializer(0.0))

	w3 = tf.get_variable("w3", shape=[hidden_nodes[1], action_dim])
	b3 = tf.get_variable("b3", shape=[1, action_dim], initializer = tf.constant_initializer(0.0))

	l1_logits = tf.matmul(state_in, w1) + b1
	l1_out = tf.tanh(l1_logits)

	l2_logits = tf.matmul(l1_logits, w2) + b2
	l2_out = tf.tanh(l2_logits)

	l3_logits = tf.matmul(l2_logits, w3) + b3
	l3_out = tf.tanh(l3_logits)

	return l3_out

# TODO: Network outputs
q_values = network(STATE_DIM, ACTION_DIM, [50,50])
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

		# place step into buffer
		buffer.append([state,action,next_state,reward,done])
		# if buffer is too long remove inital values
		if len(buffer) > MAX_SIZE_BUFFER:
			buffer.pop(0)

		# find batch size to set (to account for random.sample)
		if len(buffer) < BATCH_SIZE:
			actual_BATCH_SIZE = len(buffer)
		else:
			actual_BATCH_SIZE = BATCH_SIZE

		# collection a random subset of batches
		batchs = random.sample(buffer,actual_BATCH_SIZE)

		# create states, action and target for batch
		batch_state = []
		batch_action = []
		batch_target = []
		for batch in batchs:
			# append state
			batch_state.append(batch[0])
			# append action
			batch_action.append(batch[1])
			# find q value of next_state
			batch_next_q = q_values.eval(feed_dict={ state_in: [batch[2]] } )
			# if batch is done
			if batch[4]:
				batch_target.append(batch[3])
			else:
				batch_target.append(batch[3] + GAMMA * np.max(batch_next_q))

		# hint1: Bellman
		# hint2: consider if the episode has terminated
		# Do one training step
		session.run([optimizer], feed_dict={ state_in: batch_state,
			target_in: batch_target, action_in: batch_action
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
                                                        'Average Reward:', ave_reward, total_reward)

env.close()
