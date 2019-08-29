#!/usr/bin/env python3

import gym

from garage.experiment import run_experiment
from garage.baselines import LinearFeatureBaseline
from garage.tf.algos.go_explore import GoExplore
from garage.tf.envs import TfEnv
from garage.tf.envs import GoExploreTfEnv
from garage.tf.envs.go_explore_env import Pixel_GoExploreEnv, Ram_GoExploreEnv
from garage.tf.policies.go_explore_policy import GoExplorePolicy
from garage.tf.envs.go_explore_env import CellPool, Cell
import fire
import os
import numpy as np
from skimage.measure import block_reduce
from garage.misc.overrides import overrides
#
# class Pixel_GoExploreEnv(GoExploreTfEnv):
#     @overrides
#     def downsample(self, obs):
#         # import pdb; pdb.set_trace()
#         obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
#         obs = block_reduce(obs, block_size=(20, 20), func=np.mean)
#         obs= obs.astype(np.uint8) // 32
#         return obs.flatten()
#
# class Ram_GoExploreEnv(GoExploreTfEnv):
#     @overrides
#     def downsample(self, obs):
#         # import pdb; pdb.set_trace()
#         return obs // 32

def runner(use_ram=False,
           db_filename='/home/mkoren/Scratch/cellpool-shelf.dat',
           max_db_size=150,
           overwrite_db=True,
           n_parallel=2,
           max_path_length=2000,
           discount=0.99,
           n_itr=100,
           max_kl_step=0.01):

    if overwrite_db and os.path.exists(db_filename):
        os.remove(db_filename)

    batch_size = max_path_length * n_parallel

    def run_task(*_):
        # gym_env=gym.make('MontezumaRevenge-ram-v0')
        if use_ram:
            gym_env = gym.make('MontezumaRevenge-ram-v0')
            import pdb; pdb.set_trace()
            env = Ram_GoExploreEnv(env=gym_env)
            # env = GoExploreTfEnv(env=gym_env)
            # pool=CellPool())
            # setattr(env, 'downsampler', ram_downsampler)
        else:
            gym_env = gym.make('MontezumaRevenge-v0')
            # import pdb; pdb.set_trace()
            env = Pixel_GoExploreEnv(env=gym_env)
            # env = GoExploreTfEnv(env=gym_env)
            #                      # pool=CellPool())
            # setattr(env, 'downsampler',pixel_downsampler)

        policy = GoExplorePolicy(
            env_spec=env.spec)

        baseline = LinearFeatureBaseline(env_spec=env.spec)



        algo = GoExplore(
            db_filename=db_filename,
            max_db_size=max_db_size,
            env=env,
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            discount=discount,
            n_itr=n_itr,
            max_kl_step=max_kl_step,        )
        algo.train()


            # runner.setup(algo, env, sampler_args={'n_envs': 1})
            # runner.train(n_epochs=120, batch_size=5000,store_paths=False)


    run_experiment(
        run_task,
        snapshot_mode='last',
        seed=1,
        n_parallel=n_parallel,
    )

if __name__ == '__main__':
  fire.Fire()
