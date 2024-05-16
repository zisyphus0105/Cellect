"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces
#from gym.spaces import Special_Box

import copy
from copy import deepcopy
import numpy as np
import random
import psutil

from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import IPython
import itertools
#from eval_engines.util.core import *
import pickle
import os

import random

from circuit_design.syntehsis.synthesis_wrapper import *

#way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

class Cellect_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    #obtains yaml file
    path = os.getcwd()
    CIR_YAML = path+'/circuit_deisgn/synthesis/synthesis_inputs/yaml_files/dc.yaml'

    def __init__(self, env_config):
        self.multi_goal = env_config.get("multi_goal",False)
        self.generalize = env_config.get("generalize",False)
        num_valid = env_config.get("num_valid",50)
        self.specs_save = env_config.get("save_specs", False)
        self.valid = env_config.get("run_valid", False)

        self.env_steps = 0
        with open(Cellect_Env.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # design specs
        self.delay_boundary = yaml_data['target_specs']['delay']

        if self.generalize == False:
            specs = yaml_data['target_specs']
        else:
            load_specs_path = Cellect_Env.path+"/Cellect/gen_specs/synthesis_specs_gen_cellect"
            with open(load_specs_path, 'rb') as f:
                specs = pickle.load(f)
            
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0], reverse=True))
        if self.specs_save:
            with open("specs_"+str(num_valid)+str(random.randint(1,100000)), 'wb') as f:
                pickle.dump(self.specs, f)
        
        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.fixed_goal_idx = -1 
        self.num_os = len(list(self.specs.values())[0])
        
        yaml_params = yaml_data['params']
        
        ## define the cell numbers of every type after compaction ##
        self.yaml_param_count = [6,6,6,6]

        self.params = []
        self.param_length = []

        for value in yaml_params.values():
            param_vec = value[0].split()
            self.params.append(param_vec)
            self.param_length.append(len(param_vec))

        self.space_stop = []

        for i in range(len(self.param_length)):
            self.space_stop.append(sum(self.param_length[0:i+1]))

        self.params_id_shrink = list(yaml_params.keys())
     
        self.sim_env = synthesis_wrapper(yaml_path=Cellect_Env.CIR_YAML, num_process=1, path=Cellect_Env.path) 

        self.action_space = spaces.Tuple([spaces.Discrete(2)]*self.space_stop[-1])

        self.observation_space = spaces.Box(
            low=np.array([Cellect_Env.PERF_LOW]*2*1+self.space_stop[-1]*[1]),
            high=np.array([Cellect_Env.PERF_HIGH]*2*1+self.space_stop[-1]*[1]), dtype=np.float32)

        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)

        self.cur_params_idx = []
        for i in range(len(self.yaml_param_count)):
            self.cur_params_idx.append(np.array([j for j in range(self.yaml_param_count[i])]))

        self.global_g = []
        for spec in list(self.specs.values()):
                self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)
        self.global_g = np.array(yaml_data['normalize'])
        
        self.obj_idx = 0

    def reset(self):
        if self.generalize == True:
            if self.valid == True:
                if self.obj_idx > self.num_os-1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0,self.num_os-1)
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.g_star 
            else:
                idx = random.randint(0,self.num_os-1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)

        self.specs_ideal_norm = [self.lookup(self.specs_ideal[1], self.global_g[1])]

        self.cur_params_idx = [0] * 55

        cur_spec_norm = [0]

        reward = -10

        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])

        return self.ob
 
    def step(self, action):

        self.cur_params_idx = deepcopy(action)

        cur_spec_norm = [0]

        min_count = 0
        if (min_count < sum(action[0:self.space_stop[0]]) < self.yaml_param_count[0]) and (min_count < sum(action[self.space_stop[0]:self.space_stop[1]]) < self.yaml_param_count[1]) and (min_count < sum(action[self.space_stop[1]:self.space_stop[2]]) < self.yaml_param_count[2]) and (min_count < sum(action[self.space_stop[2]:self.space_stop[3]]) < self.yaml_param_count[3]):
            self.cur_specs = self.update(self.cur_params_idx)
            cur_spec_norm  = [self.lookup(self.cur_specs[1], self.global_g[1])]
            if self.cur_specs[0] <= self.delay_boundary[1]:
                reward = self.reward(self.cur_specs[1], self.specs_ideal[1], action)  
            else:
                reward = -5          
        else:
            reward = -10

        done = False

        if (reward >= 0):
            done = True
            print('-'*10)
            print('reward done:', reward)
            print('-'*10)
        else:
            print('+'*10)
            print('reward not done:', reward)
            print('+'*10)

        self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, action])

        self.env_steps = self.env_steps + 1

        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):

    ## calculate the performance reward ##

        spec = float(spec)
        goal_spec = float(goal_spec)
        norm_spec = (goal_spec-spec)/(goal_spec+spec)
        return norm_spec
    
    def reward(self, spec, goal_spec, action):

        ## calculate rel_specs based on the performance, library size ##
        rel_specs = 0

        if rel_specs < -0.1:
            reward = rel_specs 
        elif rel_specs < -0.15:
            reward = 5
        else:
            reward = 10

        return reward

    def update(self, params_idx):

        params_this_time = []

        index = 0

        for i in range(len(self.param_length)):
            params_this_line = []
            for j in range(self.param_length[i]):
                if params_idx[index] == 1:
                    params_this_line.append(self.params[i][j])
                index += 1

            params_this_time.append(params_this_line)
        
        param_val = [OrderedDict(list(zip(self.params_id_shrink,params_this_time)))]

        cur_specs = OrderedDict(sorted(self.sim_env.create_design_and_simulate(param_val[0])[1].items(), key=lambda k:k[0]))
        
        cur_specs = np.array(list(cur_specs.values()))
        return cur_specs

def main():
  env_config = {"generalize":True, "valid":True}
  env = Cellect_Env(env_config)
  env.reset()
  action = [0] * 55

  env.step(action)

  IPython.embed()

if __name__ == "__main__":
  main()
