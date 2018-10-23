from tidalclassifier.cnn.individual_cnn.meta_CNN import benchmarkModelDeep
from tidalclassifier.utils.helper_funcs import to_json, from_json, remove_file, read_task_id, write_list_to_file
import time
from task_array import ParallelJob
import numpy as np
from tidalclassifier.utils.combination_lock import CombinationLock

"""Grid search many hyperparameters to find the optimal (single) model/preprocessing setup

Returns:
    None
"""

class GridSearchJob(ParallelJob):
    def __init__(self, task_id, instruct_list_loc):
        ParallelJob.__init__(self, task_id, instruct_list_loc)
        self.gs_params = gridSearchParameters()
        self.aws=True
        self.run_n = self.task_id - 1 # careful, run is also a verb! Bad word choice...

    def returnCustomInstructValues(self):
        instruct = {}
        instruct['name'] = 'gs_strict'

        instruct['nb_train_samples'] = 1050
        instruct['nb_validation_samples'] = 300
        instruct['nb_epoch'] = 100
        instruct['batch_size'] = 50 #75
        instruct['confusion_images'] = 50 #75

        instruct['scale'] = 'log'
        instruct['convolve'] = False
        instruct['clip'] = False

        instruct['folds'] = 5
        instruct['runs'] = 4
        instruct['sig_n'] = 6

        instruct['aws'] = True

        # SDSS
        # instruct = {}
        # instruct['name'] = 'gs_d1'
        #
        # instruct['nb_train_samples'] = 1050
        # instruct['nb_validation_samples'] = 300
        # instruct['nb_epoch'] = 140
        # instruct['batch_size'] = 75
        # instruct['confusion_images'] = 75
        #
        # # instruct['scale'] = None
        # instruct['convolve'] = False
        # instruct['clip'] = False
        #
        # instruct['crop'] = True
        # instruct['w'] = 128
        # instruct['img_width'] = 256
        # instruct['img_height'] = 256
        #
        # instruct['folds'] = 5
        # instruct['runs'] = 4
        # instruct['sig_n'] = 6
        instruct['input_mode'] = 'threshold_3sig'

        # instruct['aws'] = 'SDSS'

        return instruct

    def createGridSearchInstructs(self):
        print('making list')
        print('gs params', self.gs_params)
        print('instruct', self.instruct)
        return generateInstructList(self.gs_params, self.instruct)

    def singleSetup(self):
        print('Single setup')
        instruct_list = self.createGridSearchInstructs()
        self.writeInstructs(instruct_list)

    def initiate(self):
        print('initiate')
        # meta needs to be re-updated
        # if self.instruct['sample_factor'] < 1:
        #     self.meta = self.meta[:int(len(self.meta)*self.instruct['sample_factor'])]
        if self.instruct['input_mode'] == 'color' or 'threshold_color_5sig' or 'threshold_color_3sig':
            print('switching to 3 channels', self.task_id)
            self.instruct['channels'] = 3
        print(self.aws)
        print(self.instruct['aws'])
        print('task', self.task_id, 'meta', len(self.meta), 'tidal', self.instruct['tidal_conf'])
        print('task', self.task_id, 'mode', self.instruct['input_mode'], 'channels', self.instruct['channels'])
        self.instruct['run_n'] = self.run_n
        benchmarkModelDeep(self.instruct, self.meta)


def generateInstructList(param_options, default_instruct):

    print('making lock')
    lock = CombinationLock(param_options)
    print('permuting lock')
    allCombinations = lock.permuteAllCombinations() # list of instruct-like dict of parameter and selected value

    # combine default_instruct with allCombinations
    instruct_list = []
    for selection_dic in allCombinations:
        modified_instruct = dict(default_instruct) # avoid reference error, should be distinct dict each time
        for key in selection_dic:
            modified_instruct[key] = selection_dic[key] # modify the default instruct dict
        instruct_list.append(modified_instruct)
    return instruct_list


def gridSearchParameters():
    gs_params = {}
    # gs_params['batch_size'] = [75,100,150,300]
    # gs_params['w'] = [50, 127]
    # gs_params['clip'] = ['ceiling','threshold']
    # gs_params['sig_n'] = [3,5,7]
    # gs_params['scale'] = 'pow'
    # gs_params['pow_val'] = [1.0,0.5,0.333]
    # gs_params['convolve'] = [True, False]
    # gs_params['rotation_range'] = [0,45]
    # gs_params['height_shift_range'] = [0,0.05]
    # gs_params['width_shift_range'] = [0,0.05]
    # gs_params['zoom_range'] = [ [1,1], [0.9,1.1] ]
    # gs_params['layer0_size'] = [32,64]
    # gs_params['layer1_size'] = [32,64]
    # gs_params['layer2_size'] = [32,64]
    # gs_params['layerFC_size'] = [32,64]
    # gs_params['tidal_conf'] = [4,34,134]
    # gs_params['sample_factor'] = [0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
    # gs_params['input_mode'] = ['stacked','threshold_5sig','threshold_bkg_5sig', 'threshold_color_5sig', 'threshold_3sig','threshold_bkg_3sig', 'threshold_color_5sig']
    # gs_params['input_mode'] = ['stacked','threshold_5sig']
    # gs_params['scale'] = ['pow','log']
    # gs_params['convolve'] = [True, False]
    # gs_params['clip'] = [True, False]

    # gs_params['input_mode'] = ['threshold_5sig','threshold_bkg_5sig', 'threshold_3sig', 'threshold_bkg_3sig_']

    # gs_params['dropout'] = [False, True]
    # manually change augmentation configuration: next, do off and False dropout


    # SDSS
    # gs_params['input_mode'] = ['stacked', 'threshold_3sig', 'threshold_5sig']
    gs_params['scale'] = [None, 'log']
    return gs_params


def gridSearch():
    task_id = read_task_id()
    index_file = 'gridsearch_index_sample'
    time.sleep(np.random.randint(4))
    job = GridSearchJob(task_id, index_file)
    print('run job')
    job.run()
    exit(0)

gridSearch()