from tidalclassifier.cnn.individual_cnn.meta_CNN import custom_flow_from_directory, create_model, fold_tables, trainCNNOnTable
from tidalclassifier.utils.helper_funcs import ThreadsafeIter, shuffle_df
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_curve, roc_auc_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def runMetaTask(instruct, run):
    print('begin meta task')
    # exit(0)
    meta = pd.read_csv('SemesterOne/meta_table.csv')

    # pick a random fold to crossValidate with, as one of n runs
    meta = shuffle_df(meta)
    folded_train_tables, folded_val_tables = fold_tables(meta, instruct)
    train_table = folded_train_tables[0]
    val_table_full = folded_val_tables[0]

    model_func = None

    print('Create data for run (0th) ' + str(run))
    createMetaclassiiferData(model_func, train_table, val_table_full, run, instruct)
    print(run, 'Exiting gracefully')
    exit(0)


def addTrueLabels(table, instruct):
    conf_4 = np.array(table.CONF == 4, dtype=bool)
    conf3 = np.array(table.CONF == 3, dtype=bool)
    conf1 = np.array(table.CONF == 1, dtype=bool)
    conf_0 = np.array(table.CONF == 0, dtype=bool)
    if instruct['tidal_conf'] == 4:
        table = table[conf_0 + conf_4]
    elif instruct['tidal_conf'] == 34:
        table = table[conf_0 + conf3 + conf_4]
    elif instruct['tidal_conf'] == 34:
        table = table[conf_0 + conf1 + conf3 + conf_4]

    table['true_label'] = table['FEAT'] != 'N'
    table['true_label'] = table['true_label'].map(lambda x: float(x)) # still autoconverts to int when extracted though...

    return table

def createMetaclassiiferData(model_func, train_table, val_table_full, run, instruct):

    instruct['run'] = run # important to set name for label record files later, must be different to not overwrite!
    # make predictions for meta train
    # split val_table into train_val and meta_val. Note that folds should be halved. Min 4.

    # add 'true label' column to the (large) train table. Pawlik will use this directly.
    # CNN generator provides true labels for large train table directly to CNN.
    # true label column in both val tables is recorded from generators in CNN stage
    # once Pawlik predictions are made, train table true labels are no longer relevant (as wtih rest of table).
    # adding here because 'instruct' is easily available
    train_table = addTrueLabels(train_table, instruct)
    val_table_full = addTrueLabels(val_table_full, instruct)

    val_table_full = shuffle_df(val_table_full) # just in case
    val_table_train = val_table_full[:int(len(val_table_full) / 2)]  # to train the meta-classifier
    val_table_test = val_table_full[int(len(val_table_full) / 2):]  # to test the meta-classifier

    val_table_train.to_csv('val_table_train.csv')
    val_table_test.to_csv('val_table_test.csv')

    print('generating CNN predictions for meta training')
    CNN_train_meta, CNN_test_meta = generateCNNPredictionTable(model_func, train_table, val_table_train, val_table_test, run,
                                                instruct)

    print('generating Pawlik predictions for meta training')
    Pawlik_train_meta, Pawlik_test_meta = generatePawlikPredictionTable(train_table, val_table_train, val_table_test, run,
                                                                        instruct)

    print(CNN_train_meta.head())
    print(Pawlik_train_meta.head())
    print(CNN_test_meta.head())
    print(Pawlik_test_meta.head())

    # inner join on picture_id to create complete meta tables
    train_meta = pd.merge(CNN_train_meta, Pawlik_train_meta, on=['picture_id','true_label'], how='inner')
    test_meta = pd.merge(CNN_test_meta, Pawlik_test_meta, on=['picture_id','true_label'], how='inner')


    # meta should already include standard_A, mask_A
    # # look up the mask values of those images in pawlik meta_table
    # if instruct['aws']: meta_loc = 'SemesterOne/meta_table.csv'
    # else: meta_loc = r'/home/mike/meta_with_A.csv'
    # meta_table_with_A = pd.read_csv(meta_loc)
    # meta_table_with_A = meta_table_with_A[['picture_id', 'standard_A', 'mask_A', 'FEAT']]  # only include pawlik data (for now)

    # inner join on picture_id to create complete meta tables
    # train_meta = pd.merge(CNN_Pawlik_train_meta, meta_table_with_A, on='picture_id', how='inner')
    # test_meta = pd.merge(CNN_Pawlik_test_meta, meta_table_with_A, on='picture_id', how='inner')

    print(train_meta.head())
    # print(train_meta['FEAT'])

    # cut non-tidal rows until there are as many tidal as non-tidal
    train_meta = evenTable(train_meta)
    test_meta = evenTable(test_meta)

    print('saving', instruct['directory'] + instruct['name'] +  '_train_meta_' + str(run) + '.csv')

    # save for later eval
    train_meta.to_csv(instruct['directory'] + instruct['name'] + '_train_meta_' + str(run) + '.csv')
    test_meta.to_csv(instruct['directory'] + instruct['name'] + '_test_meta_' + str(run) + '.csv')


def generateCNNPredictionTable(model_func, train_table, val_table_train, val_table_test, run, instruct, debug=True):
    # train the CNN, and make sure save_gen_output = True
    instruct['save_gen_output'] = False

    for network_index in range(instruct['networks']):

        trained_model = None
        model_func_string = instruct['ensemble_config']['model_func'][network_index]
        if model_func_string == 'simpleCNN': model_func = create_model
        instruct['input_mode'] = instruct['ensemble_config']['input_mode'][network_index]
        instruct['scale'] = instruct['ensemble_config']['scale'][network_index]
        print(network_index, instruct['scale'], 'network and scale')

        if network_index == 0:
            r_acc, r_val_acc, r_loss, r_val_loss, trained_model = trainCNNOnTable(model_func, train_table,
                                                                                  val_table_test, run,
                                                                                  instruct)
            networks_meta_train = predictTable(trained_model, val_table_train, instruct,
                                               network_index)  # create first table
            networks_meta_train = group_by_picture(networks_meta_train)

            networks_meta_test = predictTable(trained_model, val_table_test, instruct,
                                              network_index)  # create first table
            networks_meta_test = group_by_picture(networks_meta_test)

            if debug:
                print(networks_meta_train.head())
                print(len(networks_meta_train))

        else:

            r_acc, r_val_acc, r_loss, r_val_loss, trained_model = trainCNNOnTable(model_func, train_table,
                                                                                  val_table_test, run,
                                                                                  instruct)
            CNN_meta_train = predictTable(trained_model, val_table_train, instruct, network_index)
            CNN_meta_train = group_by_picture(CNN_meta_train)

            if debug:
                print(networks_meta_train.head())
                print(len(networks_meta_train))
            networks_meta_train = pd.merge(networks_meta_train, CNN_meta_train, on=['picture_id', 'true_label'], how='inner')
            if debug:
                print(networks_meta_train.head())
                print(len(networks_meta_train))

            CNN_meta_test = predictTable(trained_model, val_table_test, instruct, network_index)
            CNN_meta_test = group_by_picture(CNN_meta_test)
            networks_meta_test = pd.merge(networks_meta_test, CNN_meta_test, on=['picture_id', 'true_label'], how='inner')


    # grouped_train = networks_meta_train.groupby('picture_id', as_index=False)
    # grouped_train = grouped_train.aggregate(np.average)
    #
    # grouped_test = networks_meta_test.groupby('picture_id', as_index=False)
    # grouped_test = grouped_test.aggregate(np.average)

    # hypothesis: merging on duplicate key (pic id) caused tables to grow exponentially. Trying to aggregate back caused memory error OR was succesful, hiding the initial massive table

    networks_meta_train = group_by_picture(networks_meta_train)
    networks_meta_test = group_by_picture(networks_meta_test)

    return  networks_meta_train, networks_meta_test

def group_by_picture(table):
    table = table.groupby('picture_id', as_index=False)
    table = table.aggregate(np.average)
    return table

def predictTable(trained_model, input_table, instruct, network_index, debug=False):

    # pred table is the meta table supplying picture information

    # generate a prediction on each row in val_table
    instruct['save_gen_output'] = True
    temp_gen_name = 'pred' + str(np.random.randint(1000000))
    custom_gen_val = custom_flow_from_directory(input_table, instruct, even_split=True, gen_name=temp_gen_name)
    custom_gen_val = ThreadsafeIter(custom_gen_val)

    # have given generator random name to ensure no write/read errors. Should not happen in any case.
    # Predict table generators of different runs are ensured different names
    # Predict table generators of the same run in general had the same name - now tweaked.

    # make predictions on randomly augmented pred table images
    Y = np.ravel(trained_model.predict_generator(custom_gen_val, val_samples=instruct['nb_validation_samples'] * 100)) # WILL SET TO 60ish, very little augmentation averaging. Single CNN.
    # this will also cause generator output to be saved

    # load generator output
    Y_true = np.ravel(np.loadtxt(instruct['directory'] + temp_gen_name + '_' + str(instruct['run']) + '_label.txt'))[:len(Y)]  # gen overruns
    Y_pic_ids = np.ravel(np.loadtxt(instruct['directory'] + temp_gen_name + '_'  + str(instruct['run'])+ '_pic.txt'))[:len(Y)]  # gen overruns

    # place in dataframe prediction_table
    # will contain same pic ids and y_true values, but with many duplicates and in random order
    # aggregated later
    if debug:
        print('Y', Y.shape)
        print('Y_true', Y_true.shape)
        print('Y_pic_ids', Y_pic_ids)

    data = {'picture_id': Y_pic_ids, 'CNN_label_'+str(network_index): Y, 'true_label': Y_true}
    prediction_table = pd.DataFrame(data)
    return prediction_table



def generatePawlikPredictionTable(train_table, val_table_train, val_table_test, run, instruct):
    # avoid pass-by-reference errors. Will tweak tables.
    pawlik_base_train = train_table.copy()
    pawlik_train = val_table_train.copy()
    pawlik_test = val_table_test.copy()

    allowed_labels = ['standard_A', 'mask_A']
    custom_name = 'pawlik_creation'

    # train and test on even data
    pawlik_base_train = evenTable(pawlik_base_train)
    pawlik_train = evenTable(pawlik_train)
    pawlik_test = evenTable(pawlik_test)

    clf = AdaBoostClassifier()
    acc, loss, val_acc, val_loss, clf, result = benchmarkClassifierOnTables(pawlik_base_train, pawlik_train, allowed_labels, custom_name, clf)
    train_predictions = result['Y_pred']
    pawlik_train['Pawlik'] = train_predictions

    # technically only have to train once, but it's neater this way
    clf = AdaBoostClassifier()
    acc, loss, val_acc, val_loss, clf, result = benchmarkClassifierOnTables(pawlik_base_train, pawlik_test, allowed_labels, custom_name, clf)
    test_predictions = result['Y_pred']
    pawlik_test['Pawlik'] = test_predictions

    return pawlik_train, pawlik_test

def benchmarkClassifierOnTables(train_meta, test_meta, allowed_labels, custom_name, clf):

    # clf = AdaBoostClassifier()
    # clf = LogisticRegressionCV()
    acc, loss, clf = trainClassifierOnTable(clf, train_meta, allowed_labels, custom_name)
    val_acc, val_loss, clf, result = predictClassifierOnTable(clf, test_meta, allowed_labels, custom_name)

    return acc, loss, val_acc, val_loss, clf, result

def trainClassifierOnTable(clf, table, allowed_labels, custom_name, debug=True):

    allowed_data_train = table.as_matrix([allowed_labels])

    train_X = np.squeeze(allowed_data_train)
    # print(custom_name + 'train_X', train_X)

    train_Y = np.squeeze(table.as_matrix(['true_label']).astype(float))

    # print(custom_name + 'train_Y', train_Y)

    if debug:
        print('train')
        print(table.head())
        print(train_X)
        print(train_Y)

    clf.fit(train_X, train_Y)

    train_Y_pred = clf.predict_proba(train_X)[:,1]
    if debug:
        print(train_Y_pred)
        print('acc example', train_Y[0], train_Y_pred[0])
    acc = accuracy_score(train_Y.astype(int), train_Y_pred.astype(int)) # both continuous
    loss = log_loss(train_Y.astype(int), train_Y_pred.astype(int))

    return acc, loss, clf

def predictClassifierOnTable(clf, table, allowed_labels, custom_name, debug=False):

    allowed_data_test = table.as_matrix([allowed_labels])
    test_X = np.squeeze(allowed_data_test)
    test_Y = np.squeeze(table.as_matrix(['true_label']).astype(float))

    test_Y_pred = clf.predict_proba(test_X)[:,1]

    if debug:
        print('test')
        print(table.head())
        print(test_Y)
        print(test_Y_pred)
        print('acc example', test_Y[0], test_Y_pred[0])

    val_acc = accuracy_score(test_Y.astype(int), test_Y_pred.astype(int)) # both continuous

    if debug:
        print(test_Y)
        print(test_Y_pred)
        print('acc example', test_Y[0], test_Y_pred[0])

    val_loss = log_loss(test_Y, test_Y_pred)

    print(custom_name + ' val accuracy: ', val_acc)
    # print(custom_name + ' val confusion (tn, fp, fn, tp) :', np.ravel(
    #     confusion_matrix(test_Y, np.around(test_Y_pred), labels=[0, 1])))

    result = {'Y_true': test_Y, 'Y_pred': test_Y_pred}

    return val_acc, val_loss, clf, result






def evenTable(input_table):
    table = input_table.copy()
    # cut non-tidal rows until there are as many tidal as non-tidal
    # imbalance comes from tidal having far fewer unique pic ids, and grouping by pic id
    nb_tidal = len(table[table.FEAT != 'N'])
    nb_nontidal = len(table[table.FEAT == 'N'])
    while nb_nontidal > nb_tidal:
        table_inner = table[table.FEAT == 'N']
        picture_id = table_inner.iloc[np.random.randint(0,len(table_inner))]['picture_id']
        # print(picture_id)
        table = table[table.picture_id != picture_id]
        nb_nontidal = len(table[table.FEAT == 'N'])
    # print(nb_nontidal, nb_tidal)
    # print('final table length: ', len(table))
    return table

