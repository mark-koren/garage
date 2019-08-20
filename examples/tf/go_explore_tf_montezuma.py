#!/usr/bin/env python3

import gym

from garage.experiment import run_experiment
from garage.baselines import LinearFeatureBaseline
from garage.tf.algos.go_explore import GoExplore
from garage.tf.envs import TfEnv
from garage.tf.envs import GoExploreTfEnv
from garage.tf.policies.go_explore_policy import GoExplorePolicy
from garage.tf.envs.go_explore_env import CellPool, Cell

max_path_length = 2000
n_parallel = 1
batch_size = max_path_length * n_parallel

def run_task(*_):
    gym_env=gym.make('MontezumaRevenge-ram-v0')
    # import pdb; pdb.set_trace()
    env = GoExploreTfEnv(env=gym_env)
                         # pool=CellPool())

    policy = GoExplorePolicy(
        env_spec=env.spec)

    baseline = LinearFeatureBaseline(env_spec=env.spec)



    algo = GoExplore(
        env=env,
        env_spec=env.spec,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=0.99,
        max_kl_step=0.01,        )
    algo.train()


        # runner.setup(algo, env, sampler_args={'n_envs': 1})
        # runner.train(n_epochs=120, batch_size=5000,store_paths=False)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
    n_parallel=n_parallel,
)
