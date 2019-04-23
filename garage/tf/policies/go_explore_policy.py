from garage.tf.policies.base2 import StochasticPolicy2
from garage.misc.overrides import overrides
from garage.tf.distributions.diagonal_gaussian import DiagonalGaussian

class GoExplorePolicy(StochasticPolicy2):
    def __init__(self, env_spec):
        self.dist = DiagonalGaussian(dim=env_spec.action_space.flat_dim)
        self.log_std = np.zeros(env_spec.action_space.flat_dim)
        super(GoExplorePolicy, self).__init__(env_spec=env_spec, name='policy')

    # Should be implemented by all policies
    @overrides
    def get_action(self, observation):
        return self.action_space.sample(), dict()

    @overrides
    def get_actions(self, observations):
        return self.action_space.sample_n(len(observations)), dict()

    @overrides
    def get_params_internal(self, **tags):
        return []

    @overrides
    def reset(self, dones=None):
        pass

    @overrides
    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def vectorized(self):
        """
        Indicates whether the policy is vectorized. If True, it should
        implement get_actions(), and support resetting
        with multiple simultaneous states.
        """
        return True

    @overrides
    def terminate(self):
        """
        Clean up operation
        """
        pass

    @property
    def distribution(self):
        return self.dist


    def dist_info(self, obs, state_infos):
        """
        Distribution info.

        Return the distribution information about the actions.
        :param obs_var: observation values
        :param state_info_vars: a dictionary whose values should contain
         information about the state of the policy at the time it received the
         observation
        :return:
        """
        return dict(mean=None, log_std=self.log_std)
