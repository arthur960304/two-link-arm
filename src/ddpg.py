import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import pybullet
import pybulletgym.envs


np.random.seed(1000)


def weighSync(target_model, source_model, tau=0.001):
    """
    A function to soft update target networks

    : param target_model: torch object, target network
    : param source_model: torch object, source network
    : param tau: float, update factor
    """
    for param, target_param in zip(
        source_model.parameters(), target_model.parameters()
    ):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class Replay:
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        : param init_length: int, initial number of transitions to collect
        : param state_dim: int, size of the state space
        : param action_dim: int, size of the action space
        : param env: gym environment object
        """
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env

        self._storage = []
        self._init_buffer(init_length)

    def _init_buffer(self, n):
        """
        Init buffer with n samples with state-transitions taken from random actions

        : param n: int, number of samples
        """
        state = self.env.reset()
        for _ in range(n):
            action = self.env.action_space.sample()
            state_next, reward, done, _ = self.env.step(action)
            exp = {
                "state": state,
                "action": action,
                "reward": reward,
                "state_next": state_next,
                "done": done,
            }
            self._storage.append(exp)
            state = state_next

            if done:
                state = self.env.reset()
                done = False

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer

        : param exp: a dictionary consisting of state, action, reward , next state and done flag
        """
        self._storage.append(exp)
        if len(self._storage) > self.buffer_size:
            self._storage.pop(0)

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer

        : param N: int, number of samples to obtain from the buffer
        """
        return random.sample(self._storage, N)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network

        : param state_dim: int, size of the state space
        : param action_dim: int, size of the action space
        """
        super(Actor, self).__init__()

        hidden_nodes1 = 400
        hidden_nodes2 = 300
        self.fc1 = nn.Linear(state_dim, hidden_nodes1)
        self.fc2 = nn.Linear(hidden_nodes1, hidden_nodes2)
        self.fc3 = nn.Linear(hidden_nodes2, action_dim)

    def forward(self, state):
        """
        Define the forward pass of the actor

        : param state: ndarray, the state of the environment
        """
        x = torch.FloatTensor(state)

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic

        : param state_dim: int, size of the state space
        : param action_dim: int, size of the action space
        """
        super(Critic, self).__init__()

        hidden_nodes1 = 400
        hidden_nodes2 = 300
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_nodes1)
        self.fc2 = nn.Linear(hidden_nodes1, hidden_nodes2)
        self.fc3 = nn.Linear(hidden_nodes2, 1)

    def forward(self, state, action):
        """
        Define the forward pass of the critic

        : param state: ndarray, the state of the environment
        : param action: ndarray, chosen action
        """
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        x = torch.cat((state, action), dim=-1)

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class DDPG:
    def __init__(
        self,
        env,
        state_dim,
        action_dim,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        batch_size=100,
    ):
        """
        : param env: object, a gym environment
        : param state_dim: int, size of state space
        : param action_dim: int, size of action space
        : param actor_lr: float, learning rate of the actor
        : param critic_lr: float, learning rate of the critic
        : param gamma: float, discount factor
        : param batch_size: int, batch size for training
        """
        super(DDPG, self).__init__()

        self.env = env
        self.test_env = copy.deepcopy(env)  # for evaluation purpose
        self.gamma = gamma
        self.batch_size = batch_size

        # create actor networks and make sure that
        # both networks have the same initial weights
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)

        # create critic networks and make sure that
        # both networks have the same initial weights
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        # define the optim for actor and critic
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.ReplayBuffer = Replay(10000, 1000, state_dim, action_dim, env)

    def choose_action(self, state):
        """
        Select action with exploration noise
        the returned action will be clipped to [-1, 1]

        : param state: ndarray, the state of the environment
        : return: ndarray, chosen action
        """
        state = torch.FloatTensor(state)

        action = self.actor(state).data.numpy()
        action = action + np.random.normal(0, 0.1, size=self.env.action_space.shape[0])
        action = action.clip(self.env.action_space.low, self.env.action_space.high)
        return action

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

    def train(self, num_steps):
        """
        Train the policy for the given number of iterations

        :param num_steps: int, number of steps to train the policy for
        """
        actor_loss_list = []
        critic_loss_list = []
        avg_reward_list = []

        state = env.reset()
        for step in range(num_steps):
            action = self.choose_action(state)
            state_next, reward, done, _ = env.step(action)

            # store experience to replay memory
            exp = {
                "state": state,
                "action": action,
                "reward": reward,
                "state_next": state_next,
                "done": done,
            }
            self.ReplayBuffer.buffer_add(exp)

            state = state_next
            if done:
                state = env.reset()

            # sample random batch from replay memory
            exp_batch = self.ReplayBuffer.buffer_sample(self.batch_size)

            # extract batch data
            state_batch = torch.FloatTensor([exp["state"] for exp in exp_batch])
            action_batch = torch.FloatTensor([exp["action"] for exp in exp_batch])
            reward_batch = torch.FloatTensor([exp["reward"] for exp in exp_batch])
            state_next_batch = torch.FloatTensor(
                [exp["state_next"] for exp in exp_batch]
            )
            done_batch = torch.FloatTensor([1 - exp["done"] for exp in exp_batch])

            # reshape
            state_batch = state_batch.reshape(self.batch_size, -1)
            action_batch = action_batch.reshape(self.batch_size, -1)
            reward_batch = reward_batch.reshape(self.batch_size, -1)
            state_next_batch = state_next_batch.reshape(self.batch_size, -1)
            done_batch = done_batch.reshape(self.batch_size, -1)

            # pass batch data to critic network
            estimate_Q = self.critic(state_batch, action_batch)

            # pass batch data to target actor and critic network
            action_next_batch = self.actor_target(state_next_batch)
            target_Q = reward_batch + done_batch * self.gamma * self.critic_target(
                state_next_batch, action_next_batch
            )

            # calculate the critic loss and update critic
            self.optim_critic.zero_grad()
            critic_loss = F.mse_loss(estimate_Q, target_Q)
            critic_loss.backward()
            self.optim_critic.step()

            # calculate the actor loss and update actor
            self.optim_actor.zero_grad()
            actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
            actor_loss.backward()
            self.optim_actor.step()

            # soft update
            self.update_target_networks()

            # save loss data
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())

            # log
            if step % 100 == 0:
                avg_reward, count = self.eval()
                avg_reward_list.append(avg_reward)

                print(
                    "step: [{}/{}], \tactor loss: {:.4f}, \tcritic loss: {:.4f}, \taverage reward: {:.3f}, \teval steps: {}".format(
                        step, num_steps, actor_loss, critic_loss, avg_reward, count
                    )
                )
        return actor_loss_list, critic_loss_list, avg_reward_list

    def eval(self):
        """
        Evaluate the policy
        """

        count = 0
        avg_reward = 0
        done = False
        state = self.test_env.reset()

        while not done:
            action = self.actor(state).data.numpy()
            state_next, reward, done, _ = self.test_env.step(action)
            avg_reward += reward
            count += 1
            state = state_next

        avg_reward /= count
        return avg_reward, count


if __name__ == "__main__":
    # Define the environment
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)

    ddpg_object = DDPG(
        env,
        state_dim=8,
        action_dim=2,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        batch_size=100,
    )

    # Train the policy
    actor_loss, critic_loss, avg_reward = ddpg_object.train(200000)

    # save the actor network
    torch.save(ddpg_object.actor.state_dict(), "ddpg_actor.pkl")

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(actor_loss)
    plt.grid()
    plt.title("DDPG Actor Loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.savefig("ddpg_actor_loss.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(critic_loss)
    plt.grid()
    plt.title("DDPG Critic Loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.savefig("ddpg_critic_loss.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_reward)
    plt.grid()
    plt.title("DDPG Average Reward")
    plt.xlabel("*100 steps")
    plt.ylabel("average reward")
    plt.savefig("ddpg_avg_reward.png", dpi=150)
    plt.show()
