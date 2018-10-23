# plot all rocs

import os
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d

from helper_funcs import to_json, from_json



matplotlib.rcParams.update({'font.size': 32})
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 32})

def plotROCs(roc_list, fig_name):
    # expects dicts of classifier label, FPR and TPR values, interpolated to standard (0, 1, 100) linspace
    sns.set_context("poster")
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    labels = []
    for roc_data in roc_list:
        plt.plot(roc_data['fpr'], roc_data['tpr'], c=roc_data['color']) # x, y
        labels.append(roc_data['label'])
    # plt.legend(labels,loc=4)
    plt.ylabel('True Positive Rate (Completeness)')
    plt.xlabel('False Positive Rate (Contamination)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    labels.append('Random guessing')
    plt.plot(np.linspace(0, 1, 100),np.linspace(0, 1, 100), 'k--')
    plt.legend(labels,loc=4)
    plt.tight_layout()
    plt.savefig(fig_name)


def plotAUCs(roc_list, fig_loc):
    # reset
    sns.set_context("poster")
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    auc_values = [calculateAUC(roc_data) for roc_data in roc_list]
    bar_labels = [roc_data['label'] for roc_data in roc_list]
    colors = [roc_data['color'] for roc_data in roc_list]
    sns.barplot(y=bar_labels, x=np.abs(auc_values), palette=colors)
    plt.xlim([0.5, 1])
    plt.xlabel('ROC Area Under Curve (AUC)')
    plt.tight_layout()
    plt.savefig(fig_loc)


def calculateAUC(roc_data):
    y = roc_data['tpr']
    x = roc_data['fpr']
    return np.trapz(y, x)  # area under curve approximated by trapezium rule

# declare linspace global: must be same for all being averaged, but not if only plotted
# interp would be more flexible but some interps are threshold and some are fpr/tpr, gets complex

def calculateROCFromPredictions(predictions, label):
    smooth_runs = saveInterpsFromPredictions(predictions, label)
    output = calculateAverageROCFromThresholdInterps(smooth_runs)
    save_roc(output)  # saves as dict for later use without needing to recalculate
    return output

# meta benchmarks:
# for ONE classifier, interpolate FPR and TPR for ONE run, evaluate on linspace, average by run
# return standard value dict: label, fpr, tpr

# AUC is always calculated at end from np.trapz. For linspace 100, this will be quite accurate, and is much simpler.
# report will not show the raw values by run - messy


def saveInterpsFromPredictions(predictions, label):
    # results should be list of dicts (Y_true, Y_pred) for single classifier by run
    smooth_runs = {'fpr': [], 'tpr': [], 'label': label}

    for run in range(len(predictions)):
        Y_true = predictions[run]['Y_true']
        Y_pred = predictions[run]['Y_pred']
        fpr, tpr, thresholds = roc_curve(Y_true, Y_pred, pos_label=1,drop_intermediate=True)  # tidal as positive

        # flip all, so that thresholds increases for interp. (and therefore positive detections decrease)
        fpr = fpr[::-1]
        tpr = tpr[::-1]
        thresholds = thresholds[::-1]

        # manually set first element of each
        # fix 0 value at 0 positive
        fpr = np.concatenate((np.array([1]),fpr))
        tpr = np.concatenate((np.array([1]), tpr))
        thresholds = np.concatenate((np.array([0]), thresholds))

        # manually set last element of each
        # fix last value at 1 positive
        fpr = np.concatenate((fpr,np.array([0])))
        tpr = np.concatenate((tpr,np.array([0])))
        thresholds = np.concatenate((thresholds,np.array([1])))

        # print(thresholds, 'thresholds')
        # print(fpr,'fpr')
        # print(tpr,'tpr')

        smooth_run_fpr = interp1d(thresholds, fpr)
        smooth_run_tpr = interp1d(thresholds, tpr)

        smooth_runs['fpr'].append(smooth_run_fpr)
        smooth_runs['tpr'].append(smooth_run_tpr)

    return smooth_runs


def calculateAverageROCFromThresholdInterps(smoothed_funcs):
    # smoothed funcs is SINGLE CLASSIFIER interpolate functions by run for fpr, tpr, and label
    tpr_av = calculateAveragePostivesFromThresholdInterps(smoothed_funcs['tpr'])
    fpr_av = calculateAveragePostivesFromThresholdInterps(smoothed_funcs['fpr'])
    # should all return fpr and tpr values interpolated to a standard (0, 1, 100) linspace
    output = {'tpr': tpr_av, 'fpr': fpr_av, 'label': smoothed_funcs['label']}
    return output


def calculateAveragePostivesFromThresholdInterps(smoothed_funcs):
    # smoothed funcs predict ftr or tpr from thresholds
    thresholds = np.linspace(0, 1, 100)
    FPrs = np.zeros((len(smoothed_funcs),len(thresholds)))
    for func_index in range(len(smoothed_funcs)):
        func = smoothed_funcs[func_index]
        FPrs[func_index, :] = func(thresholds)
    return np.average(FPrs, axis=0) # average prediction of ftr or tpr


def findAverageCutROC(tpr_s_list, label):
    # expects list of interpolated tpr(fpr) functions of single classifier by run

    # If item is none, indicates error in evaluating full 0-1 range. See Pawlik Comparison. Quadratic max is limited.
    while tpr_s_list.count(None) != 0:
        tpr_s_list.remove(None)

    fpr = np.linspace(0, 1, num=100)
    tpr_vals = np.zeros((len(tpr_s_list),len(fpr)))
    for index in range(len(tpr_s_list)):
        smooth_func = tpr_s_list[index]
        tpr_vals[index,:] = smooth_func(fpr)
    av_tpr_vals = np.average(tpr_vals,axis=0)

    output = {'tpr':av_tpr_vals, 'fpr':fpr, 'label':label}
    save_roc(output)
    return output

####

def findCNNLabels(instruct):
    return ['CNN_label_' + str(index) for index in range(instruct['networks'])]

def resultsFromTables(instruct):
    # returns Y_true, Y_pred, by classifier by run
    cnn_labels = findCNNLabels(instruct)
    table_locs = ['/exports/eddie/scratch/s1220970/regenerated/512/' + instruct['name'] + '_train_meta_' + str(index) + '.csv' for index in range(instruct['runs'])] + \
                 ['/exports/eddie/scratch/s1220970/regenerated/512/' + instruct['name'] + '_test_meta_' + str(index) + '.csv' for index in range(instruct['runs'])]

    multi_clf_results = []
    for label_index in range(len(cnn_labels)):
        cnn_label = cnn_labels[label_index]
        results = []
        for table_loc in table_locs: # extract predictions from table
            table = pd.read_csv(table_loc)
            y_true = list(table['true_label'].values)
            y_pred = list(table[cnn_label].values)
            results.append({'Y_true': y_true, 'Y_pred': y_pred}) # standard list of dicts for one clf by run
        multi_clf_results.append(results)
    return multi_clf_results
# aggregated afterward by metabenchmarks


def plotROCsFromTables(instruct, fig_name):
    multi_clf_results = resultsFromTables(instruct)
    labels = [str(n) for n in range(len(multi_clf_results))]
    roc_list = []
    for clf_index in range(len(multi_clf_results)):
        results = multi_clf_results[clf_index]
        label = labels[clf_index]
        roc_list.append(calculateROCFromPredictions(results, label))
    plotROCs(roc_list, fig_name)
    plotROCs([roc_list[0]], fig_name[:-4]+'_single.png')
    return roc_list


def save_roc(dic):
    # expects {fpr, tpr, label}
    # save for later plotting
    # jsonify
    dic['fpr'] = dic['fpr'].tolist()
    dic['tpr'] = dic['tpr'].tolist()
    to_json(dic, dic['label'] + '_roc.txt')
    print('Saved', dic['label'])


def read_roc(data_loc, label=None, color=None):
    colors = sns.color_palette("Paired", 12)
    label_colors = {
        'Single CNN': colors[1],
        'Pawlik Regression': colors[9],
        'Pawlik Cut': colors[8],
        'WNDCHARM': colors[5],
        'Config. A': colors[7],
        'Config. B': colors[6]
    }

    roc = from_json(data_loc + '_roc.txt')
    # un-jsonify
    roc['fpr'] = np.array(roc['fpr'])
    roc['tpr'] = np.array(roc['tpr'])
    roc['label'] = label  # overwrite existing label
    roc['color'] = label_colors[label]
    print('Read', data_loc)
    return roc

if __name__ == '__main__':

    plot_dir = 'recreated_figures'

    """
    Load ROC from EDDIE run tb_m8/simple (optimal ensemble), simple (varied ensemble), and single cnn
    """
    single_cnn_roc_loc = 'data/roc_data/from_final_code_version/single_cnn/0'
    pawlik_regression_3sig_roc_loc = 'data/roc_data/from_final_code_version/pawlik/Pawlik_Regression_3sig'
    pawlik_cut_3sig_roc_loc = 'data/roc_data/from_final_code_version/pawlik/Pawlik_Cut_3sig'
    wndcharm_roc_loc = 'data/roc_data/from_final_code_version/wndcharm/WNDCHARM'  # must be calculated once beforehand
    optimal_ensemble_roc_loc = 'data/roc_data/from_final_code_version/optimal_ensemble_tb_m8/Simple'
    varied_ensemble_roc_loc = 'data/roc_data/from_final_code_version/varied_ensemble/Simple'

    if not os.path.exists(wndcharm_roc_loc):
        predictions = []
        for numeric_ID in ['000', '436']:
            y_true = np.loadtxt('data/roc_data/from_final_code_version/wndcharm/predictions/wndchrm_y_true_' + numeric_ID)
            y_score = np.loadtxt('data/roc_data/from_final_code_version/wndcharm/predictions/wndchrm_y_score_'+ numeric_ID)
            predictions.append({'Y_true': y_true, 'Y_pred':y_score}) # by run
        _ = calculateROCFromPredictions(predictions, 'WNDCHARM')  # save ROC to disk

    single_cnn_roc = read_roc(single_cnn_roc_loc, label='Single CNN')
    pawlik_regression_3sig_roc = read_roc(pawlik_regression_3sig_roc_loc, label='Pawlik Regression')
    pawlik_cut_3sig_roc = read_roc(pawlik_cut_3sig_roc_loc, label='Pawlik Cut')
    optimal_ensemble_roc = read_roc(optimal_ensemble_roc_loc, label='Config. A')
    varied_ensemble_roc = read_roc(varied_ensemble_roc_loc, label='Config. B')
    wndcharm_roc = read_roc(wndcharm_roc_loc, label='WNDCHARM')

    plotROCs([single_cnn_roc], plot_dir + '/single_roc_2018.png')

    single_v_pawlik_rocs = [
        single_cnn_roc,
        pawlik_cut_3sig_roc,
        pawlik_regression_3sig_roc]
    plotROCs(single_v_pawlik_rocs, plot_dir + '/pawlik_roc_comparison_2018.png')

    single_v_wndcharm_rocs = [single_cnn_roc, wndcharm_roc]
    plotROCs(single_v_wndcharm_rocs, plot_dir + '/wndcharm_roc_comparison_2018.png')

    cnn_rocs = [optimal_ensemble_roc, varied_ensemble_roc, single_cnn_roc]
    plotROCs(cnn_rocs, plot_dir + '/cnn_roc_ensemble_2018.png')

    """
    As above, but including wnd-charm and Pawlik (i.e. the key plot)
    """
    all_rocs = [optimal_ensemble_roc, varied_ensemble_roc, single_cnn_roc, pawlik_regression_3sig_roc, pawlik_cut_3sig_roc, wndcharm_roc]
    plotROCs(all_rocs, plot_dir + '/all_roc_final_2018.png')
    plotAUCs(all_rocs, plot_dir + '/all_auc_final_2018.png')

    """
    Deprecated - load CNN train/test predictions from run 'redo', calculate ROC, and then compare against wndcharm and any other ROCs
    """
    # predictions = []
    # for numeric_ID in ['0','1','2']:
    #     for start in ['train_', 'test_']:
    #         df = pd.read_csv('/home/mike/redo/' + start + numeric_ID + '.csv')
    #         Y_true = df['Y_true'].values
    #         Y_pred = df['Y_pred'].values
    #         predictions.append({'Y_true': Y_true, 'Y_pred': Y_pred})  # by run
    # av_cnn_roc = calculateROCFromPredictions(predictions, 'CNN')
    # plotROCs([av_wndchrm_roc] + other_rocs + [av_cnn_roc] , 'cnn_roc.png')