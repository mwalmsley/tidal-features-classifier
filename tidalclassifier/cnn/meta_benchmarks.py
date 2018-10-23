

from sklearn.metrics import accuracy_score
import numpy as np
# from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_curve, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
from matplotlib import pyplot as plt
from meta_CNN import recordMetrics
import pandas as pd
from tidalclassifier.utils.helper_funcs import from_json, calculate_metrics
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from table import benchmarkClassifierOnTables
from matplotlib import cm as cm
import seaborn as sns
from roc import plotROCs, calculateROCFromPredictions, resultsFromTables, findCNNLabels, plotROCsFromTables


def benchmarkAllOnTables(instruct):
    """
    From scratch, load subjects and train several CNN (sequentially). 
    Repeat experiment instruct['runs'] (int) times (overwritten)
    
    First, record the predictions and metrics from averaging the ensemble predictions (results_av)

    Then, record the predictions and metrics from combining predictions with logistic regression.
    Experiment with different allowed features:
        Pawlik only (results_pawlik)
        Network only (results_network)
        Both (result_combined)

    Visualise the correlation matrix between these differently-aggregated predictions
    Calculate and plot the ROC in each case.
    Also calculate standard metric figures in each case.

    Disappointingly, all the actual predictions are only state and not recorded!
    """

    instruct['runs'] = 7  # how many times to repeat the whole ensemble experiment
    runs = instruct['runs']
    instruct['run_n'] = 0

    # final benchmark measurements
    simple_acc = np.zeros(runs)
    simple_val_acc = np.zeros(runs)
    simple_loss = np.zeros(runs)
    simple_val_loss = np.zeros(runs)
    pawlik_acc = np.zeros(runs)
    pawlik_val_acc = np.zeros(runs)
    pawlik_loss = np.zeros(runs)
    pawlik_val_loss = np.zeros(runs)
    network_acc = np.zeros(runs)
    network_val_acc = np.zeros(runs)
    network_loss = np.zeros(runs)
    network_val_loss = np.zeros(runs)
    acc = np.zeros(runs)
    val_acc = np.zeros(runs)
    loss = np.zeros(runs)
    val_loss = np.zeros(runs)

    results_av = []
    results_av_both = []
    results_network = []
    results_pawlik = []
    results_pawlik_both = []
    results_combined = []

    for run in range(runs):
    # TODO: name train and test in table and meta_benchmarks with the actual name of the run?
        # load subject catalogs
        train_meta = pd.read_csv('/exports/eddie/scratch/s1220970/regenerated/512/' + instruct['name'] + '_train_meta_' + str(run) + '.csv',encoding='utf-8')
        test_meta = pd.read_csv('/exports/eddie/scratch/s1220970/regenerated/512/' + instruct['name'] + '_test_meta_' + str(run) + '.csv',encoding='utf-8')

        # record predictions and metrics for a simple average of the CNN's
        simple_acc[run], simple_loss[run], simple_val_acc[run], simple_val_loss[run], result_av, result_av_both, single_val_accs = benchmarkSimpleAverage(train_meta, test_meta, instruct)
        results_av.append(result_av)
        results_av_both.append(result_av_both)
        if run == 0:
            single_val_acc = single_val_accs
        else: single_val_acc = np.concatenate([single_val_acc, single_val_accs], axis=0)


        # using only feature_table as input, train a meta-classifier (NN? SVM? Check kaggle)
        cnn_labels = ['CNN_label_'+ str(index) for index in range(instruct['networks'])]
        pawlik_labels = ['standard_A', 'mask_A']

        clf = LogisticRegressionCV()
        allowed_labels = None
        allowed_labels = cnn_labels[:]
        network_acc[run], network_loss[run], network_val_acc[run], network_val_loss[run], network_clf, result_network = benchmarkClassifierOnTables(train_meta, test_meta, allowed_labels, 'cnn_only', clf)
        results_network.append(result_network)

        pawlik_acc[run], pawlik_loss[run], pawlik_val_acc[run], pawlik_val_loss[run], result_pawlik, result_pawlik_both = benchmarkColumn(train_meta, test_meta, 'Pawlik', instruct)
        results_pawlik.append(result_pawlik)
        results_pawlik_both.append(result_pawlik_both)

        clf = LogisticRegressionCV()
        allowed_labels = None
        allowed_labels = cnn_labels[:] + pawlik_labels[:]
        acc[run], loss[run], val_acc[run], val_loss[run], clf, result_combined = benchmarkClassifierOnTables(train_meta, test_meta, allowed_labels, 'combined', clf)
        results_combined.append(result_combined)

    labels = ['Pawlik', 'Simple', 'Network', 'Combined']
    correlationMatrix([results_pawlik, results_av, results_network, results_combined], labels, instruct['name'] + 'correlation.png')

    # calculate average roc values
    simple_roc_vals = calculateROCFromPredictions(results_av, 'Simple')
    pawlik_roc_vals = calculateROCFromPredictions(results_pawlik, 'Pawlik')
    network_roc_vals = calculateROCFromPredictions(results_network, 'Network')
    combined_roc_vals = calculateROCFromPredictions(results_combined, 'Combined')
    plt.clf()

    print(results_av)
    print(results_av_both)
    print()
    print(results_av[0])
    print(results_av_both[0])

    simple_roc_vals_both = calculateROCFromPredictions(results_av_both, 'CNN')
    # pawlik_roc_vals_both = calculateROCFromPredictions(results_pawlik_both, 'Simple')


    plotROCs([simple_roc_vals, pawlik_roc_vals, network_roc_vals, combined_roc_vals], fig_name=instruct['name'] + '_ensemble_ROCs.png')
    plt.clf()
    plotROCs([simple_roc_vals_both], fig_name=instruct['name'] + '_ensemble_ROCs_selected.png')
    visualiseMetaclassifier(single_val_acc, simple_val_acc, network_val_acc, pawlik_val_acc, val_acc, instruct)

    recordMetrics(acc,loss,val_acc, val_loss, instruct)

    pipelineCorrelationMatrix(instruct)

    single_cnn_roc_list = plotROCsFromTables(instruct, instruct['name'] + '_indiv_clf_rocs.png')
    plotROCs([single_cnn_roc_list[0],simple_roc_vals_both],instruct['name']+'_ensemble_vs_single.png')


    print('Ensemble benchmark complete, exiting gracefully')
    exit(0)


def benchmarkSimpleAverage(train_meta, test_meta, instruct):
    """Train several CNNs. Evaluate on validation subjects.
    Average predictions and compare to true labels.
    Report predictions made and metrics.
    
    Args:
        train_meta (pd.DataFrame): training catalog
        test_meta (pd.DataFrame): validation catalog
        instruct (dict): setup instructions including 'networks' (int): n of CNN to train.
    
    Returns:
        acc (np.array): accuracy from simple average train predictions vs. true labels
        loss (np.array): log loss, similarly
        val acc (np.array): similarly to acc, for validation predictions
        val loss (np.array): similarly to loss, for validation predictions
        result (dict): of form {'Y_true': true labels, Y_pred: averaged validation prediction}
        result_both (dict): similarly to result, but concatenation of both train and val predictions
        single_val_accs (np.array): 1d array of validation accuracies of each CNN
 
    """
    cnn_labels = ['CNN_label_' + str(index) for index in range(instruct['networks'])]
    # cnn_labels = ['CNN_label_' + str(index) for index in range(instruct['networks']-1)]
    train_Y = train_meta.as_matrix(['true_label'])
    train_cnn_estimates = train_meta.as_matrix([cnn_labels])
    train_Y_pred_av = np.average(train_cnn_estimates, axis=1)

    test_Y = test_meta.as_matrix(['true_label'])
    test_cnn_estimates = test_meta.as_matrix([cnn_labels])
    test_Y_pred_av = np.average(test_cnn_estimates, axis=1)

    # print single network accuracy values
    single_val_accs = []
    for label in cnn_labels:  # i.e. names of cnn
        test_Y_pred_single = test_meta[label]
        single_val_acc = accuracy_score(test_Y, np.around(test_Y_pred_single))
        print('Single CNN accuracy', label, single_val_acc)
        single_val_accs.append(single_val_acc)
    single_val_accs = np.array(single_val_accs) # 1d array of all single accuracies

    acc = accuracy_score(train_Y, np.around(train_Y_pred_av))
    val_acc = accuracy_score(test_Y, np.around(test_Y_pred_av))

    loss = log_loss(train_Y, train_Y_pred_av)
    val_loss = log_loss(test_Y, test_Y_pred_av)


    result = {'Y_true':test_Y, 'Y_pred':test_Y_pred_av}
    print(result, 'one')
    # include measurements from both tables as both tables are unseen: no metaclassifier train/test split
    result_both = {'Y_true':np.concatenate([train_Y[:], test_Y[:]]), 'Y_pred':np.concatenate([train_Y_pred_av[:], test_Y_pred_av[:]])}
    print(result_both, 'both')
    # but if using correlation matrix, don't! Need to compare same pics and same length arrays

    print('CNN simple average val accuracy: ', val_acc)

    return acc, loss, val_acc, val_loss, result, result_both, single_val_accs


def benchmarkColumn(train_meta, test_meta, label, instruct, debug=False):

    train_Y = np.squeeze(train_meta.as_matrix([label]))
    train_Y_pred = np.squeeze(train_meta.as_matrix([label]))

    test_Y = np.squeeze(test_meta.as_matrix(['true_label']))
    test_Y_pred = np.squeeze(test_meta.as_matrix([label]))

    if debug:
        print(train_Y)
        print(train_Y_pred)
        print(test_Y)
        print(train_Y_pred)

    acc, loss = calculate_metrics(train_Y, train_Y_pred)
    val_acc, val_loss = calculate_metrics(test_Y, test_Y_pred)

    result = {'Y_true':test_Y, 'Y_pred':test_Y_pred}
    # include measurements from both tables as both tables are unseen: no metaclassifier train/test split
    result_both = {'Y_true':np.concatenate([train_Y, test_Y]), 'Y_pred':np.concatenate([train_Y_pred, test_Y_pred])}
    # but if using correlation matrix, don't! Need to compare same pics and same length arrays

    print(label + ' val accuracy: ', val_acc)

    print(label, result)
    return acc, loss, val_acc, val_loss, result, result_both


def decisionFunctionToPrediction(decision_Y):
    # for LogCV http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    "The confidence score for a sample is the signed distance of that sample to the hyperplane"
    # normalise to -0.5 -> +0.5
    if decision_Y.max() > 0:
        Y = decision_Y/(2*decision_Y.max())
    else: Y = -1 * decision_Y/(2*decision_Y.max())
    Y = Y + 1
    return Y


def visualiseMetaclassifier(single, simple, network, pawlik, combined, instruct):
    plt.scatter(np.ones_like(pawlik) * 0, pawlik,c='b',marker='x')
    plt.scatter(np.ones_like(single) * 1, single, c='m', marker='x')
    plt.scatter(np.ones_like(simple) * 2, simple,c='g',marker='x')
    plt.scatter(np.ones_like(network) * 3, network,c='k',marker='x')
    plt.scatter(np.ones_like(combined) * 4, combined, c='y',marker='x')

    sz = 25
    C = 'r'
    plt.scatter(0, np.average(pawlik), sz, C, marker='x')
    plt.scatter(1, np.average(single), sz, C, marker='x')
    plt.scatter(2, np.average(simple), sz, C,marker='x')
    plt.scatter(3, np.average(network), sz, C,marker='x')
    plt.scatter(4, np.average(combined), sz, C,marker='x')

    plt.legend(['Pawlik', 'Single','Simple Av.','CNN Ensemble','Combined Ensemble'])
    # plt.title(instruct['name'])
    plt.ylabel('Validation Accuracy')
    plt.ylim([0.5, 1])
    # plt.xlabel('Simple,                       Network,                           Pawlik,                            Combined')
    plt.ylim([0, 1])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off

    plt.savefig(instruct['name'] + '_benchmark.png')

    plt.clf()
    plt.boxplot([pawlik, single, simple, network, combined],meanline=True, whis=1E6)
    plt.ylim([0.5,1])
    plt.ylabel('Validation Accuracy')
    plt.axes().set_xticklabels(['Pawlik','Single', 'CNN Avg.', 'CNN Ens.', 'Combined Ens.'])
    # plt.legend(['1. Pawlik','2. Single' '3. Simple Av.', '4. CNN Ensemble', '5. Combined Ensemble'])
    plt.savefig(instruct['name'] + '_boxplots.png')


def resultsToDataframe(results_list, labels):
    # results_list is list of list of dicts:
    # per classifier, per run, dict
    data = {}
    for classifier_index in range(len(results_list)):
        classifier_results = results_list[classifier_index]
        run_predictions = [classifier_results[run]['Y_pred'] for run in range(len(classifier_results))] # list of np arrays of predictions
        classifier_pred = np.concatenate(run_predictions) # now one long np array
        series = classifier_pred
        label = labels[classifier_index]
        print(label, len(series))
        data[label] = series
    print(data)
    df = pd.DataFrame(data)
    return df

def correlationMatrix(results_list, labels, savename):
    # results in form [ {Y_true, Y_pred} ]
    # read Y_pred into dataframe
    print(results_list)
    df = resultsToDataframe(results_list, labels)
    print(df.head())
    # visualise correlations between Y_pred
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # cmap = cm.get_cmap('jet', 30)
    # cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    # ax1.grid(True)
    # ax1.set_xticklabels(labels, fontsize=6)
    # ax1.set_yticklabels(labels, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    # fig.colorbar(cax)
    # plt.savefig(savename)
    # plt.show()

    plt.clf()
    sns.set_context("poster")
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.12, wspace=0.4)
    sns.set_style("whitegrid")

    # plot the heatmap
    sns.heatmap(df.corr(),
                xticklabels=labels,
                yticklabels=labels,
                cbar=True,
                annot=True
                      )
    plt.savefig(savename)
    plt.clf()

def pipelineCorrelationMatrix(instruct):
    # correlate each network across different tables/runs
    savename = instruct['name'] + '_pipelineCorr.png'
    # labels = ['A','B','C','D','E','F','G','H','I','J']
    # labels = instruct['ensemble_config']['input_mode']
    # labels = ['ln5sig','ln3sig','5sig','3sig','stack']
    labels = ['ln5sig', 'ln3sig', 'ln3sig', 'ln3sig', 'ln3sig']

    multi_clf_results = resultsFromTables(instruct) # dicts by clf by run

    # aggregate by run
    predictions = []
    cnn_labels = findCNNLabels(instruct)
    for cnn_index in range(len(cnn_labels)):
        all_y_pred = []
        single_clf_results = multi_clf_results[cnn_index]
        for run in range(len(single_clf_results)):
            all_y_pred = all_y_pred + list(single_clf_results[run]['Y_pred'])
        all_y_pred = np.array(all_y_pred)
        predictions.append(all_y_pred) # all predictions by clf

    data = dict(zip(labels, predictions))
    print(data)
    df = pd.DataFrame(data)
    print(df.head())

    plt.clf()
    sns.set_context("poster")
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.12, wspace=0.4)
    sns.set_style("whitegrid")

    sns.heatmap(df.corr(),
                # xticklabels=labels,
                # yticklabels=labels,
                cbar=True,
                annot=True
                      )
    plt.savefig(savename)
    plt.clf()

    # TODO: wait for last run to complete before testing
