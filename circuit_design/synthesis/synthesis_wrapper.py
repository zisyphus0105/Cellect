import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import yaml
import IPython
debug = False

class synthesis_wrapper(object):

    BASE_TMP_DIR = os.path.abspath("/tmp/dc_da")

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir == None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
            #yaml_data = yaml.load(f)
        design_netlist = yaml_data['dsn_netlist']
        design_netlist = path+'/'+design_netlist
 
        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        raw_file = open(design_netlist, 'r')
        self.tmp_lines = raw_file.readlines()
        raw_file.close()

    def get_design_name(self, state):
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def create_design(self, state, new_fname):

        design_folder = self.gen_dir+str(random.randint(0,10000))
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder + '/dc.tcl')

        ## generate tcl scripts based on the actions and specs.##

        lines = copy.deepcopy(self.tmp_lines)
        for line_num, line in enumerate(lines):

            if 'set output_path' in line:
                lines[line_num] = 'set output_path ' + design_folder

            for key, value in state.items():
                flag = 'set ' + key
                if flag in line:
                    regex = re.compile("[{](.*?[}])")
                    found = regex.search(line)
                    if found:
                        #print(lines[line_num])
                        #print(type(value[0]))
                        #print(value[0])
                        value = ' '.join(value)
                        new_replacement = "{ %s }" % value
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
                        #print(lines[line_num])

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()
        return design_folder, fpath

    def simulate(self, fpath, design_folder):
        info = 0 
        
        os.chdir(design_folder)

        ## run synthesis ##

        command = "dc_shell -64 -f %s > log " %fpath
        exit_code = os.system(command)

        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
           # raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info


    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)

        original_path = os.getcwd()

        design_folder, fpath = self.create_design(state, dsn_name)
        info = self.simulate(fpath, design_folder)
        specs = self.translate_result(design_folder)

        os.chdir(original_path)

        return state, specs, info


    def run(self, states, design_names=None, verbose=False):

        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def translate_result(self, output_path):

        circuit_result = self.parse_output(output_path)

        ## calculate delay, pdp, area out of circuit_result ##

        spec = dict(
            delay=float(delay),
            pdp=float(pdp),
            area=float(area)
        )

        return spec

    def parse_output(self, output_path):

        ## parse output reports for benchmark circuits ##
        ## please modify the following codes to get result of benchmarks ##

        circuit_set = []

        circuit_result = {}

        for circuit in circuit_set:

            input_delay = 0
            period = 0
            power = 0
            area = 0

            timing_file = output_path + '/' + circuit + '_timing.rpt'
            timing_input = open(timing_file, 'r')

            for line in timing_input.readlines():
                if 'data arrival time' in line:
                    line = line.split()
                    period = float(line[3])
                    break

                if 'input external delay ' in line:
                    line = line.split()
                    input_delay = float(line[3])

            area_file = output_path + '/' + circuit + '_area.rpt'
            area_input = open(area_file, 'r')

            for line in timing_input.readlines():
                if 'Total area' in line:
                    line = line.split()
                    area = float(line[-1])
                    break

            power_file = output_path + '/' + 'power_hier.rpt'
            power_input = open(power_file, 'r')

            for line in power_input.readlines():
                if circuit in line:
                    line = line.split()
                    power = line[-2]

            circuit_result[circuit] = ['%.2f'%(period - input_delay), power, area]

            timing_input.close()

            area_input.close()

            power_input.close()

        return circuit_result

    def find_pdp (self, circuit_result):
        ## pdp is delay * pwoer ##
        pdp = float(circuit_result[0]) * float(circuit_result[1])
        return s13207_pdp
