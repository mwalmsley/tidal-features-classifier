import json
import time
from table import runMetaTask
from tidalclassifier.utils.helper_funcs import read_task_id
from task_array import ParallelJob
import numpy as np


class MetaClassifyJob(ParallelJob):
    def __init__(self, task_id, instruct_list_loc):
        ParallelJob.__init__(self, task_id, instruct_list_loc)
        self.run_n = self.task_id - 1 # careful, run is also a verb! Bad word choice...

    def returnCustomInstructValues(self):
        instruct = {}
        instruct['name'] = 'tb_m8'
        # final is tb_m9, with augmentations off on validation images

        instruct['nb_train_samples'] = 1050
        instruct['nb_validation_samples'] = 300
        instruct['nb_epoch'] = 140
        instruct['batch_size'] = 75
        instruct['confusion_images'] = 75

        instruct['folds'] = 5
        instruct['runs'] = 10

        instruct['convolve'] = False
        instruct['clip'] = False

        instruct['dropout'] = True

        n_of_networks = 5 # linear, log, below:
        ### good different configs, x5
        # instruct['ensemble_config'] = {'model_func':['simpleCNN' for n in range(n_of_networks)],
        #                                'input_mode':['threshold_5sig','threshold_3sig','threshold_5sig','threshold_3sig','stacked'],
        #                                'scale':['log','log',None,None,None]}
        ### optimal unique config, x5
        instruct['ensemble_config'] = {'model_func':['simpleCNN' for n in range(5)],
                                       'input_mode':['threshold_3sig' for n in range(5)],
                                       'scale':['log' for n in range(5)]}
        instruct['networks'] = n_of_networks


        return instruct

    def singleSetup(self):
        self.setupMetaclassifierInstructs()

    def initiate(self):
        self.instruct['run_n'] = self.run_n
        runMetaTask(self.instruct, self.run_n)

    def setupMetaclassifierInstructs(self):
        model = None  # to ensure assignment
        runs = self.instruct['runs']
        instruct_list = [self.instruct for n in range(runs)]
        # for instruct_index in range(len(instruct_list)):
        #     if instruct_index < 3:
        #         instruct_list[instruct_index]['scale'] = 'log'
        #     else:
        #         instruct_list[instruct_index]['scale'] = None
        #     print(instruct_list[instruct_index]['scale'])
        self.writeInstructs(instruct_list)

def metaClassify():

    task_id = read_task_id()
    index_file = 'ensemble_index'
    time.sleep(np.random.randint(4))
    job = MetaClassifyJob(task_id, index_file)
    job.run()
    exit(0)

metaClassify()


