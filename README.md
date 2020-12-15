# 2 Link Arm

Policy Gradients, DDPG, and TD3 in 2 DOF arm gym env

<p align="center">
  <img width="400" height="310" src="https://github.com/arthur960304/two-link-arm/blob/main/demo.gif"/>
</p>
<p align="center">
  <em>An episode of the 2 link arm environment.</em>
</p>

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Gym Env

Download the folder modified-gym-env from [pybullet-gym-env](https://github.com/ucsdarclab/pybullet-gym-env)

### Built With

* Python 3.6.10

* PyTorch >= 1.7.0

* gym 0.17.3

* numpy >= 1.16.2

* matplotlib >= 3.1.1

## Code Organization

```
.
├── src                         # Python scripts
│   ├── policy_gradients.py     # Policy Gradients algorithm
│   ├── DDPG.py                 # Deep Deterministic Policy Gradients algorithm
│   └── TD3.py                  # Twin Delayed DDPG algorithm
├── demo.gif                    # Results
└── README.md
```

## How to Run

There are 3 methods you can try, namely policy gradients, ddpg, and td3, with corresponding file name.

ex. if you want to try policy gradients, just do
```
python policy_gradients.py
```

## Authors

* **Arthur Hsieh** - *Initial work* - [arthur960304](https://github.com/arthur960304)
