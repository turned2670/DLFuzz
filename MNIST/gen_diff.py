# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.layers import Input
from scipy.misc import imsave
from utils_tmp import *
import sys
import os
import time

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3

def load_data(path="../MNIST_data/mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

# input image dimensions
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)

model_name = sys.argv[6]

if model_name == 'model1':
    model1 = Model1(input_tensor=input_tensor)
elif model_name == 'model2':
    model1 = Model2(input_tensor=input_tensor)
elif model_name == 'model3':
    model1 = Model3(input_tensor=input_tensor)
else:
    print('please specify model name')
    os._exit(0)

print(model1.name)

# model_layer_dict1 = init_coverage_tables(model1)
model_layer_times1 = init_coverage_times(model1)  # times of each neuron covered
model_layer_times2 = init_coverage_times(model1)  # update when new image and adversarial images found
model_layer_value1 = init_coverage_value(model1)
# start gen inputs
# img_paths = image.list_pictures('../seeds_20', ext='JPEG')

img_dir = './seeds_50'
img_paths = os.listdir(img_dir)
img_num = len(img_paths)

# e.g.[0,1,2] None for neurons not covered, 0 for covered often, 1 for covered rarely, 2 for high weights
neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])
neuron_to_cover_num = int(sys.argv[3])
subdir = sys.argv[4]
iteration_times = int(sys.argv[5])

neuron_to_cover_weight = 0.5
predict_weight = 0.5
learning_step = 0.02

save_dir = './generated_inputs/' + subdir + '/'

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# start = time.clock()
total_time = 0
total_norm = 0
adversial_num = 0

total_perturb_adversial = 0

for i in xrange(img_num):

    start_time = time.clock()

    img_list = []

    img_path = os.path.join(img_dir,img_paths[i])

    img_name = img_paths[i].split('.')[0]

    mannual_label = int(img_name.split('_')[1])

    print(img_path)

    tmp_img = preprocess_image(img_path)

    orig_img = tmp_img.copy()

    img_list.append(tmp_img)

    update_coverage(tmp_img, model1, model_layer_times2, threshold)

    while len(img_list) > 0:

        gen_img = img_list[0]

        img_list.remove(gen_img)

        # first check if input already induces differences
        pred1 = model1.predict(gen_img)
        label1 = np.argmax(pred1[0])

        label_top5 = np.argsort(pred1[0])[-5:]

        update_coverage_value(gen_img, model1, model_layer_value1)
        update_coverage(gen_img, model1, model_layer_times1, threshold)

        orig_label = label1
        orig_pred = pred1

        loss_1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss_2 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-2]])
        loss_3 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-3]])
        loss_4 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-4]])
        loss_5 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-5]])

        layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)

        # neuron coverage loss
        loss_neuron = neuron_selection(model1, model_layer_times1, model_layer_value1, neuron_select_strategy,
                                       neuron_to_cover_num, threshold)
        # loss_neuron = neuron_scale(loss_neuron) # useless, and negative result

        # extreme value means the activation value for a neuron can be as high as possible ...
        EXTREME_VALUE = False
        if EXTREME_VALUE:
            neuron_to_cover_weight = 2

        layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

        # for adversarial image generation
        final_loss = K.mean(layer_output)

        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
        grads_tensor_list.extend(loss_neuron)
        grads_tensor_list.append(grads)
        # this function returns the loss and grads given the input picture

        iterate = K.function([input_tensor], grads_tensor_list)

        # we run gradient ascent for 3 steps
        for iters in xrange(iteration_times):

            loss_neuron_list = iterate([gen_img])

            perturb = loss_neuron_list[-1] * learning_step

            gen_img += perturb

            # previous accumulated neuron coverage
            previous_coverage = neuron_covered(model_layer_times1)[2]

            pred1 = model1.predict(gen_img)
            label1 = np.argmax(pred1[0])

            update_coverage(gen_img, model1, model_layer_times1, threshold) # for seed selection

            current_coverage = neuron_covered(model_layer_times1)[2]

            diff_img = gen_img - orig_img

            L2_norm = np.linalg.norm(diff_img)

            orig_L2_norm = np.linalg.norm(orig_img)

            perturb_adversial = L2_norm / orig_L2_norm

            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < 0.02:
                img_list.append(gen_img)
                # print('coverage diff = ', current_coverage - previous_coverage, 'perturb_adversial = ', perturb_adversial)

            if label1 != orig_label:
                update_coverage(gen_img, model1, model_layer_times2, threshold)

                total_norm += L2_norm

                total_perturb_adversial += perturb_adversial

                # print('L2 norm : ' + str(L2_norm))
                # print('ratio perturb = ', perturb_adversial)

                gen_img_tmp = gen_img.copy()

                gen_img_deprocessed = deprocess_image(gen_img_tmp)

                save_img = save_dir + img_name + '_' + str(get_signature()) + '.png'

                imsave(save_img, gen_img_deprocessed)

                adversial_num += 1

    end_time = time.clock()

    print('covered neurons percentage %d neurons %.3f'
          % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

    duration = end_time - start_time

    print('used time : ' + str(duration))

    total_time += duration

print('covered neurons percentage %d neurons %.3f'
      % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

print('total_time = ' + str(total_time))
print('average_norm = ' + str(total_norm / adversial_num))
print('adversial num = ' + str(adversial_num))
print('average perb adversial = ' + str(total_perturb_adversial / adversial_num))

