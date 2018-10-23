import subprocess
import time

import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
from SemesterOne.custom_image_utils import apply_corrections, get_color_image
from SemesterOne.helper_funcs import to_json, shuffle_df
from SemesterOne.meta_CNN import read_image, fold_tables, custom_flow_from_directory, define_parameters
# io.use_plugin('tifffile')
from astropy.io import fits
from skimage import exposure
from skimage.io import imsave, imshow
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

from roc import calculateROCFromPredictions, plotROCs


# def save_tifs(table, instruct, batches, folder_split=True):
#     gen = custom_flow_from_directory(table, instruct)
#     for batch in range(batches):
#         data, labels = gen.next()
#         for index in range(len(labels)):
#             label = labels[index]
#             matrix = data[index, :, :, :]
#             if matrix.min() < 0: matrix = matrix - matrix.min()
#             matrix = np.squeeze(matrix)
#             # matrix=np.expand_dims(matrix,axis=3)
#             # matrix[:,:,:,0] = np.ones_like(matrix[:,:,:,0])
#             # print(matrix.shape)
#
#             # matrix = matrix + matrix.min()
#             # print(matrix.min(),matrix.max())
#             # matrix = (255*matrix / matrix.max()).astype('uint8')
#             # print(matrix.min(),matrix.max())
#
#             # matrix = exposure.rescale_intensity(matrix.astype('float32'))
#
#             # matrix = exposure.rescale_intensity(matrix, out_range='uint8')
#             # print(matrix.min(), matrix.max())
#             # plt.imshow(matrix[:,:,0])
#             # plt.show()
#             # plt.imshow(matrix[:,:,1])
#             # plt.show()
#             # plt.imshow(matrix[:,:,2])
#             # plt.show()
#             # plt.imshow(matrix[:,:,:])
#             # plt.show()
#             print(matrix.shape)
#             print(matrix.min(), matrix.max())
#             plt.imshow(matrix)
#             plt.show()
#             # matrix = exposure.equalize_adapthist(matrix)
#             # print(matrix)
#             # im = Image.fromarray(matrix,mode='RGB')
#             im = Image.fromarray(matrix,mode='P')
#             # im.show()
#             # print(im.getbands())
#             # print(im.getcolors())
#             # print(im.getdata())
#             # print(im.getextrema())
#             # print(im.size, im.format, im.format_description)
#             # im.show()
#             # im = im.convert('L')
#             # im = im.convert('P')
#             save_str = instruct['target']
#             # print(save_str)
#             if folder_split:
#                 if labels[index] == 0:
#                     save_str += 'conf0/'
#                 else:
#                     save_str += 'conf4/'
#                 save_str += str(np.random.randint(100000)) + '.png'
#             else:
#                 # save_str += 'C' + str(label) + '_' + str(instruct['batch_size'] * batch + index) + '.tif'
#                 save_str += str(np.random.randint(100000)) + '.png'
#
#             # im_col = matrix
#             im_bw = im
#             # print(im_col.shape)
#             # print(im_bw.shape)
#             # plt.imshow(matrix[:,:,0])
#             # plt.show()
#             # print(io.find_available_plugins())
#             # imshow(im_col)
#             # imsave(save_str,im_bw)
#             # plt.show()
#             im.save(save_str)
#             time.sleep(0.01)
#             exit(0)

def save_tifs(table, instruct, batches, folder_split=True):
    gen = custom_flow_from_directory(table, instruct)
    for batch in range(batches):
        data, labels = gen.next()
        for index in range(len(labels)):
            label = labels[index]
            matrix = data[index, :, :, :]
            matrix = np.squeeze(matrix)
            im = Image.fromarray(matrix)
            im = im.convert('L')
            save_str = instruct['target']
            if folder_split:
                if labels[index] == 0:
                    save_str += 'conf0/'
                else:
                    save_str += 'conf4/'
                save_str += str(np.random.randint(100000)) + '.tif'
            else:
                save_str += 'C' + str(label) + '_' + str(instruct['batch_size'] * batch + index) + '.tif'

            im.save(save_str)
            time.sleep(0.01)

def save_for_boardgame(meta,master,instruct):
    meta = meta[meta.errors==0]
    # meta = meta[meta.ID != 'W1-85']
    # meta = meta[meta.raw_url_id.astype(int) != 71]
    # meta = meta[meta.raw_url_id.astype(int) != 72]
    # meta = meta[meta.raw_url_id.astype(int) != 73]
    for row_i in range(len(meta)):
        raw_url_id = meta.iloc[row_i]['raw_url_id']  # raw_url_id of row
        # # print(raw_url_id)
        rows = master[master.raw_url_id == raw_url_id]  # select rows with that picture_id
        # # print(rows)
        filename_list = rows['filename'].values
        # print(filename_list)
        # print(meta_i)
        if len(filename_list) != 3:
            print('warning: found ', len(filename_list)), ' files'
        else:
            im_list = [fits.getdata(instruct['directory'] + filename) for filename in filename_list]
            im = get_color_image(im_list)

            # im = np.squeeze(im)

            # im=np.expand_dims(im,axis=3)
            # im[:,:,:,0] = np.ones_like(im[:,:,:,0])
            # print(im.shape)

            im = im - im.min()
            print(im.min(), im.max())
            instruct['crop'] = True
            instruct['clip'] = True
            instruct['sig_n'] = 4
            instruct['rel'] = False
            instruct['w'] = 125
            instruct['scale'] = 'log'
            instruct['hist'] = False
            im = apply_corrections(im, instruct)
            im = np.swapaxes(im, 0, 2)
            im = (255 * im / im.max()).astype('uint8')
            # print(im.min(), im.max())

            # im = exposure.rescale_intensity(im.astype('float32'))

            # im = exposure.rescale_intensity(im, out_range='uint8')
            # print(im.min(), im.max())
            # plt.imshow(matrix[:,:,0])
            # plt.show()
            # plt.imshow(matrix[:,:,1])
            # plt.show()
            # plt.imshow(matrix[:,:,2])
            # plt.show()
            # plt.imshow(im[:,:,:])
            # plt.show()
            # matrix = exposure.equalize_adapthist(matrix)
            # print(matrix)


            if im.min() < 0: im = im - im.min()
            im = Image.fromarray(im, mode='RGB')
            # im.show()
            # print(im.getbands())
            # print(im.getcolors())
            # print(im.getdata())
            # print(im.getextrema())
            # print(im.size, im.format, im.format_description)
            # im.show()
            # im = im.convert('L')
            # im = im.convert('P')
            ID = str(rows.iloc[0]['ID'])
            feat = str(rows.iloc[0]['FEAT'])
            save_str = instruct['target'] + '_' + ID + '_' + feat + '.png'
            # print(save_str)

            im_col = im
            # im_bw = matrix[:,:,0]
            # print(im_col.shape)
            # print(im_bw.shape)
            # plt.imshow(matrix[:,:,0])
            # plt.show()
            # print(io.find_available_plugins())
            # imshow(im_col)
            imsave(save_str, im_col)
            # plt.show()
            im.save(save_str)
            time.sleep(0.01)
            exit(0)

    

def classify(table, check_feat):
    tidal_confidence_results = []
    for index in range(int(instruct['batch_size']*4)):
        image_results = classify_image(index, table, check_feat)
        if image_results != None: tidal_confidence_results.append(image_results) # [tidal conf, true label]
    return np.array(tidal_confidence_results)


def classify_image(index, table, check_feat):
    if check_feat == 'nontidal':
        class_str = 'C0.0_'
        true_label = 0
    else:
        class_str = 'C1.0_'
        true_label = 1
    classify_str = 'wndchrm classify dataset.fit ' + instruct['target'] + class_str + str(index) + '.tif'
    print(classify_str)
    output = run_wndchrm(classify_str)
    tidal_confidence = parseOutput(output)
    if tidal_confidence == None:
        return None
    else: return [tidal_confidence, true_label]



def run_wndchrm(classify_str):
    proc = subprocess.Popen(classify_str, shell=True, stdout=subprocess.PIPE)
    return proc.communicate()[0]

def parseOutput(output):
    if 'Cannot open file' in output:
        print('File missing, probably different confidence')
        return None
    else:
        substring = output[-17:-1]
        predicted_class = substring[:5]
        confidence = float(substring[-9:-1])
        if predicted_class == 'conf0':
            tidal_confidence = 1 - confidence
        else:
            tidal_confidence = confidence
        print(predicted_class)
        print(tidal_confidence)
        return tidal_confidence

# move files
aws=False
instruct, meta = define_parameters(aws)

# instruct['directory'] = r'/media/mike/SandiskLinux/512/'
# instruct['input_mode'] = 'threshold_5sig'
# instruct['input_mode'] = 'color'
# instruct['channels'] = 3

instruct['directory'] = r'/media/mike/SandiskLinux/threshold/threshold/'
meta = pd.read_csv('meta_table_wndchrm.csv')
instruct['input_mode'] = 'threshold'
instruct['channels'] = 1


# instruct['directory'] = r'/media/mike/SandiskLinux/threshold/threshold/'
# instruct['target'] = r'/home/mike/boardgame/'
instruct['target'] = r'/media/mike/SandiskLinux/augmented_2/'

# instruct['clip'] = True
instruct['scale'] = 'log'
# instruct['scale'] = None
# instruct['pow_val'] = 0.5
instruct['multiply'] = 50
# instruct['scale'] = None
# instruct['hist'] = True

instruct['w'] = 128
instruct['img_width'] = 256
instruct['img_height'] = 256

instruct['folds'] = 5

# meta = pd.read_csv('meta_table_sig.csv')
# print(meta.head())
# meta = pd.read_csv('meta_table.csv')
# meta = meta[meta.ID != 'W1-72'] # randomly missing for unknown reason

# for index, row in meta.iterrows():
#     if row.FEAT == 'N':
#         print(row['ID'], 'non-tidal',row['FEAT'])
#         command_string = 'cp ' + instruct['directory'] + row['threshold_filename'] + ' ' + instruct['target'] + 'conf0'
#     else:
#         print(row['ID'],'tidal')
#         command_string = 'cp ' + instruct['directory'] + row['threshold_filename'] + ' ' + instruct['target'] + 'conf4'
#     subprocess.call(command_string, shell=True)
#     time.sleep(0.01)

# meta = meta[:75]

# train_tables, val_tables = fold_tables(meta, instruct)
# train_table = train_tables[0]
# val_table = val_tables[0]

# rename root folder and create new empty

train_batches = 10 # / 2, per class. Appox 1000 images per batch
test_batches = 5

# meta = shuffle_df(meta)
train_table = meta[:1200]
val_table = meta[1200:]
#
train_table = meta[7:11]
val_table = meta[7:11]

# instruct['target'] = r'/media/mike/SandiskLinux/wndchrm/train/'
# save_tifs(train_table, instruct, train_batches, folder_split=True)

master = pd.read_csv('/home/mike/MPhys_Code/SemesterOne/tables/master_table.csv', sep=',')

# save_for_boardgame(meta, master, instruct)

save_tifs(val_table, instruct, test_batches, folder_split=False)

# instruct['target'] = r'/media/mike/SandiskLinux/wndchrm/train/'
# train_str = 'wndchrm train -m ' + instruct['target'] + ' ' + 'dataset.fit'
# print(train_str)
# subprocess.call(train_str, shell=True)
#
# instruct['target'] = r'/media/mike/SandiskLinux/wndchrm/test/'
# nontidal_results = classify(val_table, 'nontidal')
# print(nontidal_results)
# print(nontidal_results.shape)
# tidal_results = classify(val_table, 'tidal')
# print(tidal_results)
# print(tidal_results.shape)
# results = np.squeeze(np.concatenate([nontidal_results, tidal_results],axis=0))
# print(results)
# y_true = results[:,1]
# y_score = results[:,0]
# y_pred = np.around(results[:,0])
# #
#
# ID = str(np.random.randint(999))
# np.savetxt('wndchrm_y_true_'+ ID,y_true)
# np.savetxt('wndchrm_y_score_'+ ID, y_score)

# y_true = np.loadtxt('wndchrm_y_true')
# y_score = np.loadtxt('wndchrm_y_score')

# print(accuracy_score(y_true,y_pred))
# print('tn,fn,fp,tp')
# print(np.ravel(confusion_matrix(y_true,y_pred,labels=(0,1))))

# predictions = [{'Y_true': y_true, 'Y_pred':y_score}] # by run
#
# wndcharm_roc = calculateROCFromPredictions(predictions, 'WNDCHARM')
# plotROCs([wndcharm_roc], 'wndcharm_roc.png')

# test_str = 'wndchrm test -f0.1 -p dataset.fit report.html'
# test_str = 'wndchrm test -f0.1 -p/home/mike/Downloads/phylip-3.696 dataset.fit report_phy.html'

# test_str = 'wndchrm test -f0.2 -j295 -n5 -p /home/mike/dataset.fit report.html'
# print(test_str)
# subprocess.call(test_str, shell=True)