#!/usr/bin/env python3

import gym

from garage.experiment import LocalRunner, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos.go_explore import GoExplore
from garage.tf.envs import TfEnv
from garage.tf.policies.go_explore_policy import GoExplorePolicy

def run_task(*_):
    with LocalRunner() as runner:
        env = TfEnv(gym.make('MontezumaRevenge-ram-v0'))

        policy = GoExplorePolicy(
            env_spec=env.spec)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = GoExplore(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=200,
            discount=0.99,
            max_kl_step=0.01,
        )

        runner.setup(algo, env, sampler_args={'n_envs': 1})
        runner.train(n_epochs=120, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
