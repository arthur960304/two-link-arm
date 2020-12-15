import gym
import pybulletgym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class PolicyNet(nn.Module):
    def __init__(self, env):
        """
        Network init

        : param env: object, gym environment
        """
        super(PolicyNet, self).__init__()

        self.env = env
        num_state = self.env.observation_space.shape[0] - 1  # bug in the environment
        num_action = self.env.action_space.shape[0]

        self.fc = nn.Sequential(
            nn.Linear(num_state, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_action),
        )

        # create a covariance matrix as module parameter
        self.cov = nn.Parameter(torch.eye(2) * 0.1)

    def forward(self, x):
        """
        Feed forward

        : param x: nd array, state
        : return: tensor, probability of each action
        """
        x = torch.tensor(x, dtype=torch.float)
        output = self.fc(x)
        return output


def policy_gradient_with_baseline(
    env,
    net,
    num_epochs=200,
    batch_size=500,
    gamma=0.99,
    lr=0.001,
    enable_baseline=False,
):
    """
    Policy Gradient algorithm with baseline

    : param env: object, gym environment
    : param net: object, policy network
    : param num_epochs: int, number of epochs
    : param batch_size: int, batch size
    : param gamma: float, discount factor
    : param lr: float, learning rate
    : param enable_baseline: bool, enable baseline if True
    : return: list, average reward at each epoch
    """
    avg_reward = []
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        state = env.reset()
        reward_list = []
        estimate_reward_list = []
        batch_estimate_reward_list = []
        batch_log_prob_list = []
        reward_sum = 0
        batch_loss = 0
        log_prob_sum = 0
        num_traj = 0
        for step in range(batch_size):
            action_mean = net(state)
            action, log_prob = choose_action(net, action_mean)
            state_next, reward, done, _ = env.step(action)

            estimate_reward_list.append(reward)
            log_prob_sum += log_prob
            reward_sum += reward
            state = state_next

            if done or step == batch_size - 1:
                num_traj += 1
                batch_estimate_reward_list.append(
                    np.sum(
                        [
                            torch.pow(torch.tensor(gamma), t2 - t)
                            * estimate_reward_list[t2 - 1]
                            for t in range(1, len(estimate_reward_list) + 1)
                            for t2 in range(t, len(estimate_reward_list) + 1)
                        ]
                    )
                )
                batch_log_prob_list.append(log_prob_sum)
                reward_list.append(reward_sum)

                # reset param
                log_prob_sum = 0
                reward_sum = 0
                estimate_reward_list = []
                state = env.reset()

        avg_reward.append(np.mean(reward_list))

        if enable_baseline:
            batch_estimate_reward_list -= np.mean(batch_estimate_reward_list)

        # compute total batch loss
        batch_loss = np.sum(
            np.array(batch_estimate_reward_list) * np.array(batch_log_prob_list)
        )

        # network update
        policy_net.zero_grad()
        loss = -batch_loss / num_traj
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                "Epoch: [{}/{}],  Loss: {:.2f}, Avg reward: {:.2f}, Num of Traj: {}".format(
                    epoch + 1, num_epochs, loss, avg_reward[-1], num_traj
                )
            )
    return avg_reward


def choose_action(net, mu):
    """
    Implement stochastic policy

    : param net: object, policy network
    : param mu: tensor, mean of the multivariate normal distribution
    : return:
        action: nd array, sampled action
        log_prob: log probability of the chosen action
    """
    cov = torch.abs(net.cov) + 1e-3  # always remains positive definite
    m = MultivariateNormal(mu, cov)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.numpy(), log_prob


def plot(avg_reward, title):
    """
    Plots for average reward over every iteration

    :param avg_reward: list, a list of average reward
    :param title: str, plot title
    """
    plt.figure()
    plt.plot(avg_reward)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.savefig(title + ".png", dpi=150)
    plt.show()


if __name__ == "__main__":
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    env.reset()

    # train
    policy_net = PolicyNet(env)
    avg_reward = policy_gradient_with_baseline(
        env,
        policy_net,
        num_epochs=500,
        batch_size=1000,
        gamma=0.9,
        enable_baseline=True,
    )

    # save the model
    torch.save(policy_net.state_dict(), "model.pkl")

    # plot
    plot(avg_reward, title="Average Reward at each Iteration")
