from tidalclassifier.utils.helper_funcs import to_json, from_json, remove_file, read_task_id, write_list_to_file
import json
from tidalclassifier.cnn.individual_cnn.meta_CNN import defaultSetup
import time

class ParallelJob():

    def __init__(self, task_id, instruct_list_loc):
        self.instruct_list_loc = instruct_list_loc
        # self.aws = 'SDSS'
        self.aws = True
        self.instruct, self.meta = defaultSetup(self.aws)  # base instruct to be modified by gs_param permutations
        self.instruct['name'] = 'default_parallel'
        self.task_id = task_id
        # print('before', self.instruct)
        custom_instruct_values = self.returnCustomInstructValues()
        # print('custom', custom_instruct_values)
        self.updateInstructValues(custom_instruct_values)
        # print('after', self.instruct)

    def returnCustomInstructValues(self):
        raise NotImplementedError
        # return {}

    def singleSetup(self):
        # typically, save instruct jsons and instruct locs
        raise NotImplementedError


    def initiate(self):
        # call entry function to begin meta_CNN code
        raise NotImplementedError


    def updateInstructValues(self, new_dict):
        self.instruct.update(new_dict)


    def writeInstructs(self, instruct_list):
        instruct_locs = ['']  # first line should be blank, task id starts from 1
        grid_counter = 0
        for instruct in instruct_list:
            instruct_loc = instruct['name'] + '_' + str(grid_counter) + '_' + 'instruct.txt'
            to_json(instruct, instruct_loc)
            instruct_locs.append(instruct_loc)
            grid_counter += 1
        # write locations of instruct files to index file
        write_list_to_file(self.instruct_list_loc, instruct_locs)
        print('index written')


    def readSpecifiedInstruct(self):

        instruct_index = self.task_id

        with open(self.instruct_list_loc, 'r') as f:
            instruct_locs = f.readlines()
            instruct_loc = instruct_locs[instruct_index][:-1]  # lose the carriage return /n
        # print(instruct_loc)
        with open(instruct_loc) as json_data:
            instruct = json.load(json_data)
        return instruct


    def run(self):
        if self.task_id == 1:
            self.singleSetup()
        else:
            time.sleep(5 + self.task_id)  # stagger initiation, just in case

        self.instruct = self.readSpecifiedInstruct() # modify saved instruct
        self.initiate()

