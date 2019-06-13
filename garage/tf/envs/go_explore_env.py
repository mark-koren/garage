from garage.tf.envs.base import TfEnv
import numpy as np
from multiprocessing import Value

class CellPool():
    def __init__(self):
        print("Creating new Cell Pool:", self)
        self.pool = []
        self.guide = set()
        self.length = 0
        self.init_cell = Cell()

    def append(self, observation):
        if observation not in self.guide:
            self.guide.add(observation)
            cell = Cell()
            cell.observation = observation
            self.pool.append(cell)
            self.length += 1


    def get_cell(self, index):
        return self.pool[index]

    def get_random_cell(self):
        index = np.random.randint(0, self.length)
        return self.get_cell(index)

    def update(self, observation, trajectory, score, state):
        if observation not in self.guide:
            self.guide.add(observation)
            cell = Cell()
            cell.observation = observation
            cell.trajectory = trajectory
            cell.score = score
            cell.trajectory_length = len(trajectory)
            cell.state = state
            self.pool.append(cell)
            self.length += 1
        else:
            cell = Cell()
            cell.observation = observation
            cell = self.pool[self.pool.index(cell)]
            if score > cell.score:
                cell.score = score
                cell.trajectory = trajectory
                cell.trajectory_length = len(trajectory)
                cell.state = state
                cell.chosen_since_new = 0
            cell.seen += 1


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
        self.state = None
        self.observation = None

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if self.observation == other.observation:
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.observation))

class GoExploreTfEnv(TfEnv):
    # cell_pool = CellPool()
    pool = []
    var = Value('i', 7)
    def __init__(self, env=None, env_name=""):
        super().__init__(env, env_name)
        # self.cell_pool = cell_pool

        print("New env, pool: ", GoExploreTfEnv.pool)
        print("init object: ", GoExploreTfEnv)


    def reset(self, **kwargs):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls reset on wrapped env.
        """
        # o = super().reset(**kwargs)
        print("In reset")
        print("resetting env with value: ", GoExploreTfEnv.var.value)
        # import pdb; pdb.set_trace()
        print("resetting env with pool: ", GoExploreTfEnv.pool)
        GoExploreTfEnv.var.value = np.random.randint(0,100)
        print("resetting env with mopdified value: ", GoExploreTfEnv.var.value)
        # import pdb; pdb.set_trace
        obs = self.env.env.reset()
        # obs = self.env.env._get_obs()
        print("Got obs")
        # cell = self.cell_pool.get_random_cell()
        # if cell.state is None:
        #     return super().reset(**kwargs)
        # self.env.restore_state(cell.state)
        return obs
    #
    def step(self, action):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls step on wrapped env.
        """
        # import pdb; pdb.set_trace()
        ob, reward, done, env_info = self.env.env.step(action)
        env_info['state'] = self.env.env.clone_state()
        return ob, reward, done, env_info

    # def set_cell_pool(self, cell_pool):
    #     self.cell_pool = cell_pool
    #     print(self, "had cell pool set to: ", self.cell_pool)

    def append_cell(self, cell):
        GoExploreTfEnv.pool.append(cell)
        print("Appended Cell: ", cell, " -- pool length: ", len(GoExploreTfEnv.pool))
        print("append object: ", GoExploreTfEnv)
        print("appending env with pool: ", GoExploreTfEnv.var.value)
        GoExploreTfEnv.var.value = np.random.randint(0, 100)
        print("appending env with mopdified pool: ", GoExploreTfEnv.var.value)


    def set_param_values(self, flattened_params, **tags):
        # import pdb; pdb.set_trace()
        if tags['pool'] is not None:
            GoExploreTfEnv.pool = tags['pool']
            print("set pool")
        super().set_param_values(flattened_params, **tags)



# Might be able to speed up by having class of (observation, index) where hash and eq
# are observation, but getting obs also gets us the index
