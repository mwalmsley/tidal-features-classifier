import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper_funcs import shuffle_df, to_json, from_json
from roc import plotROCs, calculateROCFromPredictions, findAverageCutROC
from scipy.interpolate import interp1d
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from table import benchmarkClassifierOnTables, evenTable


def extractColumns(table, allowed_labels):
    allowed_data = table[allowed_labels].values
    # allowed_data = table.as_matrix([allowed_labels])
    return np.squeeze(allowed_data)

def tellme(s):
    print(s)
    # plt.title(s,fontsize=16)
    plt.draw()

def showAveragePredictionSpace(preds):
    read_pawlik(r'/home/mike/meta_with_A_full_eddie_5sig.csv') # fill background with points (different nontidals)
    nx = 50
    ny = 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    # above must match calculatePredictionSpace
    preds = np.array(preds) # expect preds as list
    av_pred = np.average(preds, axis=0)
    CS = plt.contour(x,y,av_pred)

    tellme('Use mouse to select contour label locations, middle button to finish')
    CL = plt.clabel(CS, manual=True)

    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('Mask Asymmetry') # TODO: check these are correct way around
    plt.ylabel('Standard Asymmetry')
    plt.savefig('pawlik_space.png')
    # plt.clf()

def calculatePredictionSpace(clf):
    nx = 50
    ny = 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    pred = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            # print(x,y)
            input = np.array([x[i], y[j]])
            # print(input)
            input.reshape(1, -1)
            # print(input.shape)
            pred[i, j] = clf.predict_proba(input)[:, 1]
    return pred

def convertPawlikToFeatures(even=True):
    meta_full = pd.read_csv('/home/mike/meta_with_A.csv')

    # doesn't generalise to 34, 134
    conf_4 = np.array(meta_full.CONF == 4, dtype=bool)
    conf_0 = np.array(meta_full.CONF == 0, dtype=bool)
    meta = meta_full[conf_0 + conf_4]

    meta = evenTable(meta)

    meta['true_label'] = meta['FEAT'] != 'N'
    meta['true_label'] = meta['true_label'].map(lambda x: float(x))
    # x = extractColumns(meta,['standard_A','mask_A'])
    # y_true_str = extractColumns(meta, ['FEAT'])
    # y_true = (y_true_str != 'N').astype(float)
    # print(x)
    # print(y_true)
    # return standard_A_4, standard_A_0, mask_A_4, mask_A_0

    return meta[['standard_A','mask_A','true_label']]
    # return x, y_true

def findInterpolatedCutROC(train, test):
# fit polynomial using train sample
# calculate fpr, tpr on test sample for all cuts using that polynomial
# interpolate fpr, tpr into smooth line
    standard_A = train['standard_A']
    mask_A = train['mask_A']

    z = np.polyfit(standard_A, mask_A, deg=2)
    p = np.poly1d(z)

    standard_A_array = np.linspace(-0.1,1.1,num=200)

    fpr = np.zeros_like(standard_A_array)
    tpr = np.zeros_like(standard_A_array)
    # print(test.head())
    for index in range(len(standard_A_array)):
        tpr[index], fpr[index] = sampleCut(test, standard_A_array[index], p(standard_A_array[index]))
    print(fpr.min())
    # plt.plot(standard_A_array, p(standard_A_array))
    if fpr.min() > 0.0 or fpr.max() < 1.0:
        print(fpr)
        print(tpr)
        print(standard_A_array)
        print(p(standard_A_array))
        # plt.plot(standard_A_array, p(standard_A_array),'k--')
        # plt.show()
        print('Minima/maxima error. Skipping')
        return None
        # plt.plot(standard_A_array, p(standard_A_array))
        # plt.xlim([0,1])
        # plt.ylim([0,1])
        # plt.show()
        # exit(0)
    tpr_smooth = interp1d(fpr, tpr)
    return tpr_smooth

def sampleCut(table, standard_cut, mask_cut):
    # print(table.head())
    # tn = len(table[(table.FEAT == 'N') & (table.standard_A < standard_cut) & (table.mask_A < mask_cut)])
    fp = len(table[(table.true_label == 0) & ( (table.standard_A > standard_cut) | (table.mask_A > mask_cut))])
    # fn = len(table[(table.FEAT != 'N') & (table.standard_A < standard_cut) & (table.mask_A < mask_cut)])
    tp =len(table[(table.true_label == 1) & ((table.standard_A > standard_cut) | (table.mask_A > mask_cut))])

    p = len(table[table.true_label == 1])
    n = len(table[table.true_label == 0])

    tpr = float(tp) / float(p)
    fpr = float(fp) / float(n)

    return tpr, fpr

def read_pawlik(fname):
    meta_full = pd.read_csv(fname)

    conf0 = meta_full.CONF == 0
    conf4 = meta_full.CONF == 4
    meta_full = meta_full[conf0 + conf4]

    meta_full = evenTable(meta_full)

    standard_A_4 = meta_full[meta_full.FEAT != 'N']['standard_A']
    standard_A_0 = meta_full[meta_full.FEAT == 'N']['standard_A']
    mask_A_4 = meta_full[meta_full.FEAT != 'N']['mask_A']
    mask_A_0 = meta_full[meta_full.FEAT == 'N']['mask_A']

    standard_A = meta_full['standard_A']
    mask_A = meta_full['mask_A']

    z = np.polyfit(standard_A,mask_A,deg=2)
    p = np.poly1d(z)
    x = np.linspace(0,1,num=100)
    # # print(standard_A_4)
    plt.scatter(standard_A_4, mask_A_4, color='r')
    # plt.scatter(standard_A_0, mask_A_0, color='k', alpha=0.5)
    plt.scatter(standard_A_0, mask_A_0, color='k')
    plt.plot(x,p(x),'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(['fit' ,'tidal', 'non'], loc=0)
    # plt.legend(['tidal'], loc=0)
    plt.xlabel('Asymmetry w/ Rmax')
    plt.ylabel('Mask Asymmetry w/ Rmax')
    plt.savefig('pawlik_scatter_1727_3sig.png')

    #
    # plt.hold(1)
    #
    #
    # standard_N = meta_full[meta_full.FEAT == 'N']['standard_A']
    # mask_N = meta_full[meta_full.FEAT == 'N']['mask_A']
    #
    #
    # standard_A = meta_full[meta_full.FEAT == 'A']['standard_A']
    # mask_A = meta_full[meta_full.FEAT == 'A']['mask_A']
    #
    # standard_F = meta_full[meta_full.FEAT == 'F']['standard_A']
    # mask_F = meta_full[meta_full.FEAT == 'F']['mask_A']
    #
    #
    # standard_M = meta_full[meta_full.FEAT == 'M']['standard_A']
    # mask_M = meta_full[meta_full.FEAT == 'M']['mask_A']
    #
    #
    # standard_L = meta_full[meta_full.FEAT == 'L']['standard_A']
    # mask_L = meta_full[meta_full.FEAT == 'L']['mask_A']
    #
    # standard_S = meta_full[meta_full.FEAT == 'S']['standard_A']
    # mask_S = meta_full[meta_full.FEAT == 'S']['mask_A']
    #
    #
    # standard_H = meta_full[meta_full.FEAT == 'H']['standard_A']
    # mask_H = meta_full[meta_full.FEAT == 'H']['mask_A']
    #
    #
    # alpha = 0.8
    # # plt.scatter(standard_N, mask_N,color='k',alpha=alpha)
    # plt.scatter(standard_A, mask_A,color='g',alpha=alpha)
    # plt.scatter(standard_F, mask_F, color='r',alpha=alpha)
    # plt.scatter(standard_M, mask_M,color='b',alpha=alpha)
    # plt.scatter(standard_L, mask_L,color='m',alpha=alpha)
    # plt.scatter(standard_S, mask_S,color='y',alpha=alpha)
    # plt.scatter(standard_H, mask_H,color='c',alpha=alpha)
    #
    #
    #
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # # plt.legend(['N', 'A', 'F', 'M', 'L', 'S', 'H'], loc=0)
    # plt.legend(['A', 'F', 'M', 'L', 'S', 'H'], loc=0)
    # plt.xlabel('Asymmetry w/ Rmax')
    # plt.ylabel('Mask Asymmetry w/ Rmax')
    # plt.savefig('pawlik_scatter_1000_3sig_FEAT_tidal.png')
    #
    # alpha = 0.8
    # # plt.scatter(standard_N, mask_N,color='k',alpha=alpha)
    # plt.scatter(standard_A, mask_A,color='g',alpha=alpha)
    # plt.scatter(standard_F, mask_F, color='r',alpha=alpha)
    # plt.scatter(standard_M, mask_M,color='b',alpha=alpha)
    # plt.scatter(standard_L, mask_L,color='m',alpha=alpha)
    # plt.scatter(standard_S, mask_S,color='y',alpha=alpha)
    # plt.scatter(standard_H, mask_H,color='c',alpha=alpha)
    #
    #
    #
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.legend(['N', 'A', 'F', 'M', 'L', 'S', 'H'], loc=0)
    # plt.xlabel('Asymmetry w/ Rmax')
    # plt.ylabel('Mask Asymmetry w/ Rmax')
    # plt.savefig('pawlik_scatter_1000_3sig_tidal.png')
    #
    #
    #
    #
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.legend(['tidal', 'non'], loc=0)
    # plt.xlabel('Asymmetry w/ Rmax')
    # plt.ylabel('Mask Asymmetry w/ Rmax')
    # plt.savefig('pawlik_scatter_1000_3sig.png')



folds = 5
runs = 200
predictions = []
mesh_predictions = []
tpr_s_list = []
# fpr_s_list = []
for run in range(runs):
    feats = convertPawlikToFeatures(even=False)
    # print(feats.head())

    separator = int(len(feats) * ((folds - 1.0) / float(folds)))  # 1.0, float to avoid early rounding
    # print(len(feats))
    # print(separator)
    feats = shuffle_df(feats)
    train = feats.iloc[:separator]
    test = feats.iloc[separator:]

    # print(train.head())
    # print(test.head())

    # cut process follows quite a different pattern to with classifier
    # outside loop, average interpolated lines for final ROC plot

    tpr_s = findInterpolatedCutROC(train, test)
    tpr_s_list.append(tpr_s)

    clf = LogisticRegressionCV()
    acc, loss, val_acc, val_loss, clf, prediction = benchmarkClassifierOnTables(train, test, ['mask_A', 'standard_A'],
                                                                            'pawlik',clf)
    predictions.append(prediction)
    # preds.append(calculatePredictionSpace(clf))

regression_roc = calculateROCFromPredictions(predictions, 'Pawlik_Regression_5sig')
# showAveragePredictionSpace(mesh_predictions) # interactive, save manually
cut_roc = findAverageCutROC(tpr_s_list, 'Pawlik_Cut_5sig')

#
# print('jsonify')
#
# for dic in cut_roc:
#     print(dic['label'])
#     dic['fpr'] = dic['fpr'].tolist()
#     dic['tpr'] = dic['tpr'].tolist()
# for dic in clf_roc:
#     print(dic['label'])
#     dic['fpr'] = dic['fpr'].tolist()
#     dic['tpr'] = dic['tpr'].tolist()
#     # print(dic['tpr'])
#     # print(dic['tpr'].max())
# to_json(clf_roc, 'clf_roc_3sig.txt')
# to_json(cut_roc, 'cut_roc_3sig.txt')
#
# clf_roc_3sig = from_json('clf_roc_3sig.txt')
# # clf_roc_5sig = from_json('clf_roc_5sig.txt')
# cut_roc_3sig = from_json('cut_roc_3sig.txt')
# # cut_roc_3sig = from_json('cut_roc_3sig.txt')
#
# # un-jsonify
# for dic in cut_roc_3sig:
#     print(dic['label'])
#     dic['fpr'] = np.array(dic['fpr'])
#     dic['tpr'] = np.array(dic['tpr'])
#
# for dic in clf_roc_3sig:
#     dic['fpr'] = np.array(dic['fpr'])
#     dic['tpr'] = np.array(dic['tpr'])

# data_list = cut_roc_3sig + clf_roc_3sig

plotROCs([regression_roc, cut_roc], 'Pawlik_ROCs.png')

