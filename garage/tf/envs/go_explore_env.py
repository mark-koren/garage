from garage.tf.envs.base import TfEnv
import numpy as np
from multiprocessing import Value
from garage.misc.overrides import overrides
import pdb
import random
import shelve
from bsddb3 import db
import pickle
import time



class CellPool():
    def __init__(self, filename = 'database.dat', flag=db.DB_RDONLY, flag2='r'):
        # print("Creating new Cell Pool:", self)
        # self.guide = set()

        # import pdb; pdb.set_trace()
        # self.pool = [self.init_cell]
        # self.guide = self.init_cell.observation
        self.length = 1

        # self.d_pool = {}

        pool_DB = db.DB()
        # print('Creating Cell Pool with flag:', flag)
        # print(filename)
        pool_DB.open(filename, dbname=None, dbtype=db.DB_HASH, flags=flag)
        # pool_DB = None
        self.d_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
        self.key_list = []
        self.max_value = 0
        # self.d_pool = shelve.BsdDbShelf(pool_DB)
        # self.d_pool = shelve.open('/home/mkoren/Scratch/cellpool-shelf2', flag=flag2)
        # self.d_pool = shelve.DbfilenameShelf('/home/mkoren/Scratch/cellpool-shelf2', flag=flag2)

    def create(self):

        self.init_cell = Cell()
        self.init_cell.observation = np.zeros((1,128))
        self.init_cell.trajectory = None
        self.init_cell.score = -np.inf
        self.init_cell.state = None
        # self.d_pool = shelve.open('cellpool-shelf', flag=flag)

        self.d_pool[str(hash(self.init_cell))] = self.init_cell
        self.key_list.append(str(hash(self.init_cell)))
        self.length = 1
        # import pdb; pdb.set_trace()

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
            self.key_list.append(obs_hash)
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

    @property
    def fitness(self):
        return max(1, self.score)

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
        self.params_set = False
        self.db_filename = 'database.dat'
        self.key_list = []
        self.max_value = 0
        # self.downsampler = self.default_downsampler
        super().__init__(env, env_name)
        # self.cell_pool = cell_pool

        # print("New env, pool: ", GoExploreTfEnv.pool)
        # print("New env: ", self, " test_var: ", self.test_var)
        # print("init object: ", GoExploreTfEnv)

    def sample(self, population):
        attempts = 0
        while attempts < 10000:
            candidate = population[random.choice(self.p_key_list.value)]
            if random.random() < (candidate.score / self.max_value):
                return candidate
        print("Returning Uniform Random Sample - Max Attempts Reached!")
        return population[random.choice(self.p_key_list.value)]



    def reset(self, **kwargs):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls reset on wrapped env.
        """
        # print("reset")
        # print(self.p_db_filename.value)
        # flag = db.DB_RDONLY
        # pool_DB = db.DB()
        # pool_DB.open(self.p_db_filename.value, dbname=None, dbtype=db.DB_HASH, flags=flag)
        # dd_pool = shelve.Shelf(pool_DB)
        # cell = dd_pool[random.choice(list(dd_pool.keys()))]
        # dd_pool.close()
        # pool_DB.close()
        # pdb.set_trace()
        # self.env.env.restore_state(cell.state)
        # obs = self.env.env._get_obs()
        try:
            # pdb.set_trace()
            # start = time.time()
            flag = db.DB_RDONLY
            pool_DB = db.DB()
            # tick1 = time.time()
            pool_DB.open(self.p_db_filename.value, dbname=None, dbtype=db.DB_HASH, flags=flag)
            # tick2 = time.time()
            dd_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
            # tick3 = time.time()
            # keys = dd_pool.keys()
            # tick4_1 = time.time()
            # list_of_keys = list(keys)
            # tick4_2 = time.time()
            # choice = random.choice(self.p_key_list.value)
            # import pdb; pdb.set_trace()
            # tick4_3 = time.time()
            # cell = dd_pool[choice]
            cell = self.sample(dd_pool)
            # tick5 = time.time()
            dd_pool.close()
            # tick6 = time.time()
            pool_DB.close()
            # tick7 = time.time()
            # print("Make DB: ", 100*(tick1 - start)/(tick7 - start), " %")
            # print("Open DB: ", 100*(tick2 - tick1) / (tick7 - start), " %")
            # print("Open Shelf: ", 100*(tick4_2 - tick2) / (tick7 - start), " %")
            # # print("Get all keys: ", 100*(tick4_1 - tick3) / (tick7 - start), " %")
            # # print("Make list of all keys: ", 100 * (tick4_2 - tick4_1) / (tick7 - start), " %")
            # print("Choose random cell: ", 100 * (tick4_3 - tick4_2) / (tick7 - start), " %")
            # print("Get random cell: ", 100*(tick5 - tick4_3) / (tick7 - start), " %")
            # print("Close shelf: ", 100*(tick6 - tick5) / (tick7 - start), " %")
            # print("Close DB: ", 100*(tick7 - tick6) / (tick7 - start), " %")
            # print("DB Access took: ", time.time() - start, " s")
            if cell.state is not None:
                if cell.state[0] == 0:
                    print("DEFORMED CELL STATE")
                    obs = self.env.env.reset()
                else:
                    # print("restore state: ", cell.state)
                    self.env.env.restore_state(cell.state)
                    # print("restored")
                    obs = self.env.env._get_obs()
                    # print("restore obs: ", obs)
            else:
                print("Reset from start")
                obs = self.env.env.reset()
            # pdb.set_trace()
        except db.DBBusyError:
            print("DBBusyError")
            obs = self.env.env.reset()
        except db.DBLockNotGrantedError or db.DBLockDeadlockError:
            print("db.DBLockNotGrantedError or db.DBLockDeadlockError")
            obs = self.env.env.reset()
        except db.DBForeignConflictError:
            print("DBForeignConflictError")
            obs = self.env.env.reset()
        except db.DBAccessError:
            print("DBAccessError")
            obs = self.env.env.reset()
        except db.DBPermissionsError:
            print("DBPermissionsError")
            obs = self.env.env.reset()
        except db.DBNoSuchFileError:
            print("DBNoSuchFileError")
            obs = self.env.env.reset()
        except db.DBError:
            print("DBError")
            obs = self.env.env.reset()
        except:
            print("Failed to get state from database")
            obs = self.env.env.reset()

        # o = super().reset(**kwargs)
        # print("In reset")
        # if self.p_pool is not None: print("Cell Pool Length: ", self.p_pool.get_value().length)

        #TODO: This will be slow, need to implement faster random item dict (memory trade off)
        #TODO: Also need to sample cells proportional to reward
        #https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights
        # cell = self.p_pool.value.d_pool[random.choice(list(self.p_pool.value.d_pool.keys()))]
        # if cell.state is not None:
        #     # pdb.set_trace()
        #     self.env.env.restore_state(cell.state)
        #     obs = self.env.env._get_obs()
        # else:
        #     obs = self.env.env.reset()
        # obs = self.env.env._get_obs()
        # print("Got obs")
        # cell = self.cell_pool.get_random_cell()
        # if cell.state is None:
        #     return super().reset(**kwargs)
        # self.env.restore_state(cell.state)
        # print("downsample")
        x = self.downsample(obs)
        # print("return from reset")
        return x
    #
    def step(self, action):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls step on wrapped env.
        """
        # import pdb; pdb.set_trace()
        # print("step")
        try:
            obs, reward, done, env_info = self.env.env.step(action)
            env_info['state'] = self.env.env.clone_state()
            if env_info['state'][0] == 0:
                print("GOT DEFORMED STATE: ", obs, reward, action, done)
                import sys; sys.exit()
            return self.downsample(obs), reward, done, env_info
        except:
            pdb.set_trace()

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
            # self.p_pool = GoExploreParameter("pool", CellPool())
            self.p_db_filename = GoExploreParameter("db_filename", self.db_filename)
            self.p_key_list = GoExploreParameter("key_list", self.key_list)
            # self.p_downsampler = GoExploreParameter("downsampler", self.downsampler)
            # self.p_var = GoExploreParameter("var", GoExploreTfEnv.var)
            # self.p_test_var = GoExploreParameter("test_var", self.test_var)
            self.params_set = True

        if tags.pop("db_filename", False) == True:
            return [self.p_db_filename]

        if tags.pop("key_list", False) == True:
            return [self.p_key_list]

        # if tags.pop("downsampler", False) == True:
            # return [self.p_downsampler]

        return [self.p_db_filename, self.p_key_list]#, self.p_downsampler]

        # if tags.pop("pool", False) == True:
        #     return [self.p_pool]
        # if tags.pop("test_var", False) == True:
            # return [self.p_test_var]
        # if tags.pop("pool", False) == True:
        #     return [self.p_pool]
        # return [self.p_pool]
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

    @overrides
    def get_param_values(self, **tags):
        return [
            param.get_value(borrow=True) for param in self.get_params(**tags)
        ]

    def downsample(self, obs):
        return obs

    # def downsample(self, obs):
    #     return self.downsampler(obs=obs)
    #
    # def default_downsampler(self, obs):
    #     print("DEFAULT DOWNSAMPLE")
    #     return obs
from skimage.measure import block_reduce

class Pixel_GoExploreEnv(GoExploreTfEnv):
    @overrides
    def downsample(self, obs):
        # import pdb; pdb.set_trace()
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        obs = block_reduce(obs, block_size=(20, 20), func=np.mean)
        obs= obs.astype(np.uint8) // 32
        return obs.flatten()

class Ram_GoExploreEnv(GoExploreTfEnv):
    @overrides
    def downsample(self, obs):
        # import pdb; pdb.set_trace()
        return obs // 32


# Might be able to speed up by having class of (observation, index) where hash and eq
# are observation, but getting obs also gets us the index
