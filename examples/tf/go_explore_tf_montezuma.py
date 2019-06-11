#!/usr/bin/env python3

import gym

from garage.experiment import run_experiment
from garage.baselines import LinearFeatureBaseline
from garage.tf.algos.go_explore import GoExplore
from garage.tf.envs import TfEnv
from garage.tf.envs import GoExploreTfEnv
from garage.tf.policies.go_explore_policy import GoExplorePolicy
from garage.tf.envs.go_explore_env import CellPool, Cell

def run_task(*_):
    env = GoExploreTfEnv(env=gym.make('MontezumaRevenge-ram-v0'))
                         # pool=CellPool())

    policy = GoExplorePolicy(
        env_spec=env.spec)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = GoExplore(
        env=env,
        env_spec=env.spec,
        policy=policy,
        baseline=baseline,
        max_path_length=200,
        discount=0.99,
        max_kl_step=0.01,        )
    algo.train()


        # runner.setup(algo, env, sampler_args={'n_envs': 1})
        # runner.train(n_epochs=120, batch_size=5000,store_paths=False)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
    n_parallel=8,
)
