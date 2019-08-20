from garage.tf.envs.base import TfEnv
import numpy as np
from multiprocessing import Value
from garage.misc.overrides import overrides
import pdb
import random
import shelve
from bsddb3 import db
class CellPool():
    def __init__(self, flag=db.DB_RDONLY):
        print("Creating new Cell Pool:", self)
        # self.guide = set()

        # import pdb; pdb.set_trace()
        # self.pool = [self.init_cell]
        # self.guide = self.init_cell.observation
        self.length = 1

        # self.d_pool = {}

        pool_DB = db.DB()
        print('Creating Cell Pool with flag:', flag)
        pool_DB.open('cellpool-shelf.dat', dbname=None, dbtype=db.DB_HASH, flags=flag)
        self.d_pool = shelve.Shelf(pool_DB)

    def create(self):

        self.init_cell = Cell()
        self.init_cell.observation = np.zeros((1,128))
        self.init_cell.trajectory = None
        self.init_cell.score = -np.inf
        self.init_cell.state = None
        # self.d_pool = shelve.open('cellpool-shelf', flag=flag)

        self.d_pool[str(hash(self.init_cell))] = self.init_cell
        self.length = 1

    def append(self, cell):
        # pdb.set_trace()
        # if observation not in self.guide:
        #     self.guide.add(observation)
        #     cell = Cell()
        #     cell.observation = observation
        #     self.pool.append(cell)
        #     self.length += 1
        if cell in self.d_pool:
            self.d_pool[cell].seen += 1
        else:
            self.d_pool[cell] = cell


    def get_cell(self, index):
        return self.pool[index]

    def get_random_cell(self):
        index = np.random.randint(0, self.length)
        return self.get_cell(index)

    def update(self, observation, trajectory, score, state):
        # pdb.set_trace()
        cell = Cell()
        cell.observation = observation
        #This tests to see if the observation is already in the matrix
        if not np.any(np.equal(observation, self.guide).all(1)):
            # self.guide.add(observation)
            self.guide = np.append(self.guide, np.expand_dims(observation, axis=0), axis = 0)
            cell.trajectory = trajectory
            cell.score = score
            cell.trajectory_length = len(trajectory)
            cell.state = state
            self.pool.append(cell)
            self.length += 1
            return True
        else:
            # cell = Cell()
            # cell.observation = observation
            cell = self.pool[self.pool.index(cell)]
            if score > cell.score:
                cell.score = score
                cell.trajectory = trajectory
                cell.trajectory_length = len(trajectory)
                cell.state = state
                cell.chosen_since_new = 0
            cell.seen += 1
        return False

    def d_update(self, observation, trajectory, score, state):
        # pdb.set_trace()
        #This tests to see if the observation is already in the matrix
        obs_hash = str(hash(observation.tostring()))
        if not obs_hash in self.d_pool:
            # self.guide.add(observation)
            cell = Cell()
            cell.observation = observation
            # self.guide = np.append(self.guide, np.expand_dims(observation, axis=0), axis = 0)
            cell.trajectory = trajectory
            cell.score = score
            cell.trajectory_length = len(trajectory)
            cell.state = state
            self.d_pool[obs_hash] = cell
            self.length += 1
            return True
        else:
            cell = self.d_pool[obs_hash]
            if score > cell.score:
                cell.score = score
                cell.trajectory = trajectory
                cell.trajectory_length = len(trajectory)
                cell.state = state
                cell.chosen_since_new = 0
            cell.seen += 1
            self.d_pool[obs_hash] = cell
        return False


class Cell():

    def __init__(self):
        # print("Creating new Cell:", self)
        # Number of times this was chosen and seen
        self.seen=0
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
        if np.all(self.observation == other.observation):
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.observation.tostring()))

class GoExploreParameter():
    def __init__(self, name, value, **tags):
        self.name = name
        self.value = value
        # pdb.set_trace()

    def get_value(self, **kwargs):
        return self.value

    def set_value(self, value):
        self.value = value




class GoExploreTfEnv(TfEnv):
    # cell_pool = CellPool()
    # pool = []
    # var = Value('i', 7)
    def __init__(self, env=None, env_name=""):
        self.test_var = 6
        self.p_pool = None
        self.params_set = False
        super().__init__(env, env_name)
        # self.cell_pool = cell_pool

        # print("New env, pool: ", GoExploreTfEnv.pool)
        # print("New env: ", self, " test_var: ", self.test_var)
        # print("init object: ", GoExploreTfEnv)


    def reset(self, **kwargs):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls reset on wrapped env.
        """
        # o = super().reset(**kwargs)
        # print("In reset")
        # if self.p_pool is not None: print("Cell Pool Length: ", self.p_pool.get_value().length)

        #TODO: This will be slow, need to implement faster random item dict (memory trade off)
        #TODO: Also need to sample cells proportional to reward
        #https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights
        cell = self.p_pool.value.d_pool[random.choice(list(self.p_pool.value.d_pool.keys()))]
        if cell.state is not None:
            # pdb.set_trace()
            self.env.env.restore_state(cell.state)
            obs = self.env.env._get_obs()
        else:
            obs = self.env.env.reset()
        # obs = self.env.env._get_obs()
        # print("Got obs")
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

    # def append_cell(self, cell):
        # GoExploreTfEnv.pool.append(cell)
        # print("Appended Cell: ", cell, " -- pool length: ", len(GoExploreTfEnv.pool))
        # print("append object: ", GoExploreTfEnv)
        # print("appending env with pool: ", GoExploreTfEnv.var.value)
        # GoExploreTfEnv.var.value = np.random.randint(0, 100)
        # print("appending env with mopdified pool: ", GoExploreTfEnv.var.value)

    @overrides
    def get_params_internal(self, **tags):
        # this lasagne function also returns all var below the passed layers
        if not self.params_set:
            self.p_pool = GoExploreParameter("pool", CellPool())
            # self.p_var = GoExploreParameter("var", GoExploreTfEnv.var)
            # self.p_test_var = GoExploreParameter("test_var", self.test_var)
            self.params_set = True

        # if tags.pop("pool", False) == True:
        #     return [self.p_pool]
        # if tags.pop("test_var", False) == True:
            # return [self.p_test_var]
        if tags.pop("pool", False) == True:
            return [self.p_pool]
        return [self.p_pool]
        # return [self.p_pool,self.p_var, self.p_test_var]

    @overrides
    def set_param_values(self, param_values, **tags):
        # pdb.set_trace()
        # if tags['pool'] is not None:
        #     GoExploreTfEnv.pool = tags['pool']
        #     print("set pool")

        debug = tags.pop("debug", False)
        # param_values = unflatten_tensors(flattened_params,
        #                                  self.get_param_shapes(**tags))
        for param,  value in zip(
            self.get_params(**tags),
            param_values):
            param.set_value(value)
            if debug:
                print("setting value of %s" % param.name)
        # super().set_param_values(flattened_params, **tags)



# Might be able to speed up by having class of (observation, index) where hash and eq
# are observation, but getting obs also gets us the index
