# def simple_train(model, meta_train, meta_test, fresh_model=False):
#     """given data table, fit the data to model"""
#     time.sleep(0.5) # wait half a second to allow any generators to complete
#     if fresh_model: # reinstantiate
#         config = model.get_config()
#         model = Sequential.from_config(config)
#
#     custom_gen_train = custom_flow_from_directory(train_table, directory, custom_batch, even_split=True,
#                                                       input_mode=input_mode)
#     custom_gen_val = custom_flow_from_directory(val_table, directory, custom_batch, even_split=True,
#                                                     input_mode=input_mode)
#     custom_gen_train = threadsafe_iter(custom_gen_train)
#     custom_gen_val = threadsafe_iter(custom_gen_val)
#
#     hist = model.fit_generator(
#         custom_gen_train,
#         samples_per_epoch=nb_train_samples,
#         nb_epoch=nb_epoch,
#         validation_data=custom_gen_val,
#         nb_val_samples=nb_validation_samples,
#     )
#
#     return model
import matplotlib.pyplot as plt
import numpy as np
from meta_CNN import defaultSetup, crossValidationRandom, create_model, construct_image
from helper_funcs import write_fits, read_task_id
from custom_image_utils import estimate_background

def heatmaps(models, epochs, new_weights):
    pic_id = read_task_id()
    if pic_id == 1: pic_id = 0
    aws = True
    n_heatmap_models = models
    heatmap_epochs = epochs
    print('heatmap epochs', heatmap_epochs)

    savename_core = 'heatmap_bkg_' + str(pic_id)

    instruct, meta = defaultSetup(aws=aws)
    if aws: instruct['input_mode'] = 'threshold_bkg_3sig'
    else: instruct['input_mode'] = 'threshold'
    instruct['folds'] = 5
    instruct['name'] = 'heatmap'

    instruct['nb_train_samples'] = 1050
    instruct['nb_validation_samples'] = 75
    instruct['crop'] = True
    instruct['w'] = 64 # minimum crop width due to network architecture
    if instruct['crop'] == True: instruct['img_width'] = instruct['w'] * 2
    if instruct['crop'] == True: instruct['img_height'] = instruct['w'] * 2

    chosen_picture_id = pic_id
    print('pic_id', pic_id)
    print('models', models)
    print('epochs', epochs)
    print('new_weights', new_weights)
    row = np.squeeze(meta[meta.picture_id == chosen_picture_id])
    meta = meta[meta.picture_id != chosen_picture_id] # don't train on heatmap image (though should be minor)
    img = construct_image(row, instruct, augmentations=False)

    write_fits(img, instruct['directory'], '', savename_core + '_img.fits')

    heatmap_list = []
    for heatmap_index in range(len(heatmap_epochs)):
        instruct['nb_epoch'] = heatmap_epochs[heatmap_index]
        savename = savename_core + '_' + str(heatmap_index) + '.fits'
        heatmap = calculateHeatmap(img, n_heatmap_models, savename, instruct, meta, new=new_weights)
        heatmap = heatmap - heatmap[0,0] # plot deviation from unobstructed image
        write_fits(heatmap, instruct['directory'], '', savename) # write to root
        heatmap_list.append(heatmap)
    heatmap_diff_12 = heatmap_list[1] - heatmap_list[0]
    heatmap_diff_32 = heatmap_list[2] - heatmap_list[1]
    heatmap_diff_31 = heatmap_list[2] - heatmap_list[0]
    write_fits(heatmap_diff_12, instruct['directory'], '', savename_core + '_diff_12.fits')  # write to root
    write_fits(heatmap_diff_32, instruct['directory'], '', savename_core + '_diff_32.fits')  # write to root
    write_fits(heatmap_diff_31, instruct['directory'], '', savename_core + '_diff_31.fits')  # write to root

    print('heatmap success')
    exit(0)

def calculateHeatmap(img,  n_heatmap_models, savename, instruct, meta, new=False):
    # train n models, save weights
    models = ['error' for n in range(n_heatmap_models)]
    if new:
        instruct['runs'] = 1
        for model_n in range(len(models)):
            acc, loss, val_acc, val_loss, model = crossValidationRandom(create_model, meta, instruct)
            model.save_weights(instruct['input_mode'] +'_'+ str(model_n) + '_' + str(instruct['nb_epoch']) + '.h5') # both index and epochs must match
            models[model_n] = model
    else:
        for model_n in range(len(models)):
            models[model_n] = create_model(instruct)
            models[model_n].load_weights(instruct['input_mode'] +'_'+ str(model_n) + '_' + str(instruct['nb_epoch']) + '.h5') # both index and epochs must match

    heatmap = occlusion_2D(img, models)
    return heatmap


def occlusion_2D(img, trained_models):
    n_models = len(trained_models)
    # print(model_n)
    # box_length = 3
    # box_length = 10
    img_width_x = np.shape(img)[1]
    img_width_y = np.shape(img)[2]
    predictions = np.zeros((n_models, img_width_x, img_width_y))
    # print(np.shape(predictions))
    for model_n in range(n_models):
        print('model: ', model_n)
        for pixel_x in range(img_width_x):
            print(pixel_x)
            for pixel_y in range(img_width_y):
                box_center = (pixel_x, pixel_y)
                inner_img = np.copy(img) # will eidt inner_img. Avoid pass-by-reference.
                input_data = np.expand_dims(inner_img,axis=0)
                if mode == 'occlude':
                    input_data[0,:,:,:] = box_occluded_image(inner_img, box_center, box_length)
                else:
                    flat_bkg, fake_bkg, bkg_std = estimate_background(inner_img)
                    bkg_val = flat_bkg[0,0]
                    if mode == 'paint':
                        input_data[0, :, :, :] = box_paint_image(inner_img, box_center, box_length, bkg_val, bkg_std)
                    elif mode == 'background':
                        input_data[0, :, :, :] = box_background_image(inner_img, box_center, box_length, bkg_val, bkg_std)
                predictions[model_n, pixel_x, pixel_y] = trained_models[model_n].predict(input_data)

    print(np.shape(predictions))
    print(predictions)

    ensemble_predictions = np.average(predictions, axis=0)

    print(ensemble_predictions)
    print(np.shape(ensemble_predictions))
    # plt.imshow(ensemble_predictions)
    # plt.colorbar()
    # plt.show()
    return ensemble_predictions

def box_occluded_image(img, box_center, box_length):
    box_center_x, box_center_y = box_center[0], box_center[1]
    img_channels, img_x, img_y = np.shape(img)
    # for all pixels, occlude if within box
    for i in range(box_center_x, box_center_x + box_length + 1):
        for j in range(box_center_y, box_center_y + box_length + 1):
            if i >= 0 and i < img_x:
                if j >= 0 and j < img_y:
                    img[0, i, j] = np.min(img)
    # if box_center == (100,100):
    #     plt.imshow(img[0, :, :])
    #     plt.show()
    return img

def box_paint_image(img, box_center, box_length, bkg_mean, bkg_std):
    box_center_x, box_center_y = box_center[0], box_center[1]
    img_channels, img_x, img_y = np.shape(img)
    # for all pixels, fill with structure if within box
    for i in range(box_center_x, box_center_x + box_length + 1):
        for j in range(box_center_y, box_center_y + box_length + 1):
            if i >= 0 and i < img_x:
                if j >= 0 and j < img_y:
                    img[0, i, j] = np.random.normal(bkg_mean+bkg_std*4, bkg_std) # fill with 4sigma 'structure'
    # if box_center == (100,100):
    #     plt.imshow(img[0, :, :])
    #     plt.show()
    return img

def box_background_image(img, box_center, box_length, bkg_mean, bkg_std):
    box_center_x, box_center_y = box_center[0], box_center[1]
    img_channels, img_x, img_y = np.shape(img)
    # for all pixels, fill with structure if within box
    for i in range(box_center_x, box_center_x + box_length + 1):
        for j in range(box_center_y, box_center_y + box_length + 1):
            if i >= 0 and i < img_x:
                if j >= 0 and j < img_y:
                    img[0, i, j] = np.random.normal(bkg_mean, bkg_std) # fill with fake background
    # if box_center == (100,100):
    #     plt.imshow(img[0, :, :])
    #     plt.show()
    return img

# pic_id = 17
models = 3
epochs = [3,15,120]
box_length = 3 # passed as global
mode = 'background' # passed as global
new_weights = False
heatmaps(models, epochs, new_weights)


    # # for 1:1 occlusion method: (better to package model list into occlusion method?)
# def occlusion_1D(selected_img, selected_label, trained_model):
#     n_models = len(trained_model)
#     # print(model_n)
#     width = 10
#     inner_threshold_list = np.arange(0, 140, width)
#     predictions = np.zeros((n_models, len(inner_threshold_list)))
#     for model_n in range(n_models):
#         for index in range(len(inner_threshold_list)):
#             inner_img = np.copy(selected_img)
#             inner_threshold = inner_threshold_list[index]
#             inner_img[0, :, :, :] = radial_occluded_image(inner_img[0], inner_threshold, width)
#             predictions[model_n, index] = trained_model[model_n].predict(inner_img)
#     ensemble_predictions = np.average(predictions, axis=0)
#     print(ensemble_predictions)
#     if selected_label == 0: accuracy = 1 - ensemble_predictions
#     if selected_label == 1: accuracy = ensemble_predictions
#     plt.plot(inner_threshold_list, accuracy)
#     plt.xlabel('Inner ring radii')
#     plt.ylabel('Accuracy of tidal prediction')
#     plt.show()
#     heatmap_1D(selected_img[0], inner_threshold_list, width, ensemble_predictions)
#
#
# # acc, loss, val_acc, val_loss, model = crossValidation(InceptionV3, meta, instruct, 3)
# def radial_occluded_image(img, inner_threshold, width):
#     img_x, img_y = np.shape(img[0])
#     center_x, center_y = img_x / 2, img_y / 2
#     # annulus
#     for i in range(img_x):
#         for j in range(img_y):
#             radius = np.linalg.norm((center_x - i, center_y - j))
#             if radius > inner_threshold and radius < (inner_threshold + width): img[0, i, j] = np.min(img)
#     # plt.imshow(img[0,:,:])
#     plt.show()
#     return img
#
#
# def heatmap_1D(img, inner_threshold_list, width, preds):
#     # print(np.shape(img))
#     img = img * 0
#     img_x, img_y = np.shape(img[0])
#     center_x, center_y = img_x / 2, img_y / 2
#     for index in range(len(inner_threshold_list)):
#         inner_threshold = inner_threshold_list[index]
#         for i in range(img_x):
#             for j in range(img_y):
#                 radius = np.linalg.norm((center_x - i, center_y - j))
#                 if radius > inner_threshold and radius < (inner_threshold + width): img[0, i, j] = preds[index]
#     plt.imshow(img[0, :, :])
#     plt.colorbar()
#     plt.show()
#     return img