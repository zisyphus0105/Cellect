import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from cellect.envs.action_observation import Cellect_Env

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
args = parser.parse_args()
ray.init()

#configures training of the agent with associated hyperparameters
#See Ray documentation for details on each parameter

#Horizon: Number of steps after which the episode is forced to terminate

config_train = {
            #"sample_batch_size": 200,
            "train_batch_size": 400,
            #"sgd_minibatch_size": 1200,
            #"num_sgd_iter": 3,
            #"lr":1e-3,
            #"vf_loss_coeff": 0.5,
            "horizon":  16,
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [128, 64, 32, 16]},
            "num_workers": 12,
            "env_config":{"generalize":True, "run_valid":False},
            }


if not args.checkpoint_dir:
    trials = tune.run_experiments({
        "cellect_on_synthesis": {
        "checkpoint_freq":1,
        "run": "PPO",
        "env": Cellect_Env,
        "stop": {"episode_reward_mean": -0.02},
        "config": config_train},
    })
else:
    print("RESTORING NOW!!!!!!")
    tune.run_experiments({
        "restore_ppo": {
        "run": "PPO",
        "config": config_train,
        "env": Cellect_Env,
        "restore": args.checkpoint_dir,
        "checkpoint_freq":1},
    })
