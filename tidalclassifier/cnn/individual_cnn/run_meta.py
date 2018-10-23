"""
Simple deep benchmark of single defined model over many random splits - useful for debugging
Does not use any ensembling.
"""
import logging
import os
import json
import argparse

from tidalclassifier.cnn.individual_cnn import meta_CNN
from tidalclassifier.cnn import metric_utils, input_utils
from tidalclassifier.general_metrics import performance_by_class


def runBenchmark(name, model_dir, aws=False, test_mode=True, cv_mode='exhaustive'):
    instruct, catalog = meta_CNN.defaultSetup(aws=aws)  # default run paramers, includes directory setup
    instruct['name'] = name
    instruct['model_dir'] = model_dir

    # uses stacked images by default
    instruct['input_mode'] = 'threshold_3sig'

    if test_mode:
        instruct['nb_train_samples'] = 75
        instruct['nb_validation_samples'] = 75
        instruct['nb_epoch'] = 3
    else:  # full run, day-scale on CPU only, hour-scale on AWS
        instruct['nb_train_samples'] = 1050
        instruct['nb_validation_samples'] = 300
        instruct['nb_epoch'] = 250

    instruct['batch_size'] = 75

    model_func = meta_CNN.create_model  # remember, callable to get a new model, not a model itself
    # train, save metrics to disk
    if cv_mode == 'exhaustive':
        instruct['folds'] = 5
        instruct['runs'] = 5  # must have 1 exactly run per fold - we train on each fold
        train_tables, val_tables = input_utils.fold_tables(catalog, instruct) # exhaustive folds
    elif cv_mode == 'random':
        instruct['folds'] = 5
        instruct['runs'] = 1  # can have arbitrarily many runs
        train_tables, val_tables = meta_CNN.get_random_folds(catalog, instruct)  # random folds

    logging.info(len(train_tables))
    logging.info(len(train_tables[0]))
    logging.info(len(val_tables))
    logging.info(len(val_tables[0]))

    meta_CNN.trainCNNOnTables(model_func, train_tables, val_tables, instruct)

    logging.info('Benchmark complete. Exiting gracefully')


def show_metrics(model_dir):
    with open(os.path.join(model_dir, 'instruct.json'), 'r') as f:
        instruct = json.load(f)
    metrics = metric_utils.load_metrics_as_table(instruct)
    metric_utils.plot_aggregate_metrics(metrics, instruct['model_dir'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cross-validate a CNN')
    parser.add_argument('--name', dest='name', type=str,
                    help='Name of CNN experiment')
    parser.add_argument('--results_dir', dest='results_dir', type=str,
                    help='Directory into which to place experiment results folder')
    parser.add_argument('--cv_mode', dest='cv_mode', type=str,
                    help='Which folds to train on. Either exhaustive or random')
    # bools which are true if provided (e.g. --aws) and false otherwise
    parser.add_argument('--aws', dest='aws', default=False, action='store_true',
                    help='Use directory structure of AWS Deep Learning AMI instance')
    parser.add_argument('--test_mode', dest='test_mode', default=False, action='store_true',
                    help='Run minimal training to test execution')
    args = parser.parse_args()

    model_dir = os.path.join(args.results_dir, args.name)
    # place all results here
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)


    logging.basicConfig(
        filename=os.path.join(model_dir, 'run.log'),
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )
    logging.info('Args: {}'.format(args))

    runBenchmark(args.name, model_dir, aws=args.aws, test_mode=args.test_mode, cv_mode=args.cv_mode)
    show_metrics(model_dir)
    performance_by_class.get_performance(model_dir)

# python tidalclassifier/cnn/individual_cnn/run_meta.py --name stacked_local_test --results_dir results/cnn_runs --cv_mode=random --test_mode