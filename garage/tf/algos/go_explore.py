"""Natural Policy Gradient Optimization."""
from enum import Enum, unique

import numpy as np
import tensorflow as tf

# from garage.logger import logger, tabular
from garage.misc import special
from garage.misc.overrides import overrides
from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.misc import tensor_utils
from garage.tf.misc.tensor_utils import compute_advantages
from garage.tf.misc.tensor_utils import discounted_returns
from garage.tf.misc.tensor_utils import filter_valids
from garage.tf.misc.tensor_utils import filter_valids_dict
from garage.tf.misc.tensor_utils import flatten_batch
from garage.tf.misc.tensor_utils import flatten_batch_dict
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.optimizers import LbfgsOptimizer
from garage.tf.envs.go_explore_env import GoExploreTfEnv

import pdb

class GoExplore(BatchPolopt):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo,
    etc.
    """

    def __init__(self,
                 env,
                 env_spec,
                 policy,
                 **kwargs):
        # env,
        # policy,
        # baseline,
        # scope = None,
        # n_itr = +500,
        # start_itr = 0,
        # batch_size = 5000,
        # max_path_length = 500,
        # discount = 0.99,
        # gae_lambda = 1,
        # plot = False,
        # pause_for_plot = False,
        # center_adv = True,
        # positive_adv = False,
        # store_paths = False,
        # whole_paths = True,
        # fixed_horizon = False,
        # sampler_cls = None,
        # sampler_args = None,
        # force_batch_sampler = False,
        """
        :param env_spec: Environment specification.
        :type env_spec: EnvSpec
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if
         running multiple algorithms
        simultaneously, each using different environments and policies
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param center_adv: Whether to rescale the advantages so that they have
         mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are
         always positive. When used in conjunction with center_adv the
         advantages will be standardized before shifting.
        :return:
        """
        self.env_spec = env_spec
        self.policy = policy
        self.env = env

        # self.init_opt()

        super().__init__(env=env,
                         policy=policy, **kwargs)

    @overrides
    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        # pdb.set_trace()
        # self.cell_pool = CellPool()
        # self.cell_pool.append(Cell())
        # self.policy.set_param_values({"cell_num":-1,
        #                               "stateful_num":-1,
        #                               "cell_pool": self.cell_pool})
        # self.policy.set_cell_pool(self.cell_pool)
        # self.env.set_cell_pool(self.cell_pool)
        # GoExploreTfEnv.pool.append(Cell())
        self.env.append_cell(Cell())
        self.env.set_param_values({'pool': self.env.pool})


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        return {'env':None, 'paths':None}

    @overrides
    def optimize_policy(self, itr, samples_data):
        self.env.append_cell(Cell())
        self.env.set_param_values({'pool':self.env.pool})
        # self.policy.set_param_values({"cell_num": -1,
        #                               "stateful_num": itr,
        #                               "cell_pool": self.cell_pool})
        # self.env.


        pdb.set_trace()


class CellPool():
    def __init__(self):
        print("Creating new Cell Pool:", self)
        self.pool = []
        self.length = 0

    def append(self, cell):
        self.pool.append(cell)
        self.length += 1

    def get_cell(self, index):
        return self.pool[index]

class Cell():

    def __init__(self):
        print("Creating new Cell:", self)
        # Number of times this was chosen and seen
        self.seen_times = []
        self.chosen_times = 0
        self.chosen_since_new = 0
        self.score = -np.inf
        self.action_times = 0

        self.trajectory_length = -np.inf
        self.trajectory = []

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        # if self.trajectory

    # @dataclass
    # class Cell:
    #     # The list of ChainLink that can take us to this place
    #     chain: typing.List[ChainLink] = copyfield([])
    #     seen: list = copyfield({})
    #     score: int = -infinity
    #
    #     seen_times: int = 0
    #     chosen_times: int = 0
    #     chosen_since_new: int = 0
    #     action_times: int = 0  # This is the number of action that led to this cell
    #     # Length of the trajectory
    #     trajectory_len: int = infinity
    #     # Saved restore state. In a purely deterministic environment,
    #     # this allows us to fast-forward to the end state instead
    #     # of replaying.
    #     restore: typing.Any = None
    #     # TODO: JH: This should not refer to a Montezuma-only data-structure
    #     exact_pos: MontezumaPosLevel = None
    #     trajectory: list = copyfield([])
    #     real_cell: MontezumaPosLevel = None
