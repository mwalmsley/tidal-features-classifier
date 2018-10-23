import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(color_codes=True)


# results_n = 12
# file_locs = ['/home/mike/gs_s2/gs_s2_' + str(n) + '_acc_results.txt' for n in range(results_n)]


# file_list = [0,3,6,9]


def plot_results(results_n, file_locs):
    final_accs_av = np.zeros(results_n)
    final_accs_std = np.zeros(results_n)

    for file_index in range(results_n):
        file_loc = file_locs[file_index]
        data = np.loadtxt(file_loc)
        print(data.shape)
        data_av = np.average(data, 0)

        # plt.subplot(211)
        # plt.plot(data.transpose())
        # plt.ylim(0.5, 1)
        #
        # plt.subplot(212)
        # plt.ylim(0.5, 1)
        # plt.plot(np.arange(data.shape[1]),data_av)
        # plt.show()

        final_acc = data[:, -1]
        final_acc_av = np.average(final_acc)
        final_acc_std = np.std(final_acc)

        final_accs_av[file_index] = final_acc_av
        final_accs_std[file_index] = final_acc_std

    plt.errorbar(np.arange(results_n), final_accs_av, yerr=final_accs_std * 1)


def plotAccuracyCurve(name):

    sns.set_context("poster")
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.12, wspace=0.4)
    sns.set_style("whitegrid")

    acc_loc = name+'_acc_results.txt'
    acc = np.loadtxt(acc_loc)

    print(acc.shape)
    val_acc_loc = name + '_val_acc_results.txt'
    val_acc = np.loadtxt(val_acc_loc)

    plt.clf()
    if type(acc[0]) != np.float64:
        epochs = len(acc[0])
    else:
        epochs = len(acc)
    tick_spacing = 5
    x_ticks = np.arange(0,epochs+tick_spacing, tick_spacing)

    plt.figure(1)
    plt.subplot(121)
    sns.tsplot(val_acc, ci=95)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.4, 1.0])
    plt.xticks(x_ticks)
    plt.title('Validation')


    plt.subplot(122)
    sns.tsplot(acc, ci=95, color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.4, 1.0])
    plt.xticks(x_ticks)
    # plt.axes().set_xticks(np.arange(0,len(acc[0])+25,25))
    # plt.legend(['Train set', 'Validation set'], loc=0)
    plt.title('Training')

    plt.savefig(name + '_acc.png')


# results_n = 4
#
# file_list = [0,3,6,9]
# file_locs = ['/home/mike/gs_p2/gs_p2_' + str(n) + '_acc_results.txt' for n in file_list]
# plot_results(results_n, file_locs)
#
# plt.legend(['conf4','conf34','conf134'])
# plt.ylabel('Validation accuracy')
# plt.axes().set_xticklabels(['','0.25','0.5','0.75','1.0'])
# plt.xlabel('Sample fraction')
# plt.xlim([-1,results_n])
# plt.show()


# file_list = [0,3,6,9]
# file_locs = ['/home/mike/gs_p2/gs_p2_' + str(n) + '_acc_results.txt' for n in file_list]

# name = '/home/mike/gs_p2/gs_p2_10'
# name = '/home/mike/16th Feb/grid_search/aug/aug_0'
# name = '/home/mike/16th Feb/grid_search/aug/aug_9'
# name = '/home/mike/16th Feb/grid_search/aug/aug_0'
name = '/home/mike/vgg_l3/vgg_l3_top_0'
plotAccuracyCurve(name)

