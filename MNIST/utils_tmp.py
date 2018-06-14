# -*- coding: utf-8 -*-

import random
from collections import defaultdict

import numpy as np
from datetime import datetime
from keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image

model_layer_weights_top_k = []

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data


def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


def decode_label(pred):
    return decode_predictions(pred)[0][0][1]


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 1e4 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3

def init_coverage_tables(model1):
    model_layer_dict1 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    return model_layer_dict1


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False

def init_coverage_times(model):
    model_layer_times = defaultdict(int)
    init_times(model,model_layer_times)
    return model_layer_times

def init_coverage_value(model):
    model_layer_value = defaultdict(float)
    init_times(model, model_layer_value)
    return model_layer_value

def init_times(model,model_layer_times):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_times[(layer.name, index)] = 0

def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_to_cover(not_covered,model_layer_dict):
    if not_covered:
        layer_name, index = random.choice(not_covered)
        not_covered.remove((layer_name, index))
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index

def random_strategy(model,model_layer_times, neuron_to_cover_num):
    loss_neuron = []
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_times.items() if v == 0]
    for _ in xrange(neuron_to_cover_num):
        layer_name, index = neuron_to_cover(not_covered, model_layer_times)
        loss00_neuron = K.mean(model.get_layer(layer_name).output[..., index])
        # if loss_neuron == 0:
        #     loss_neuron = loss00_neuron
        # else:
        #     loss_neuron += loss00_neuron
        # loss_neuron += loss1_neuron
        loss_neuron.append(loss00_neuron)
    return loss_neuron

def neuron_select_high_weight(model, layer_names, top_k):
    global model_layer_weights_top_k
    model_layer_weights_dict = {}
    for layer_name in layer_names:
        weights = model.get_layer(layer_name).get_weights()
        if len(weights) <= 0:
            continue
        w = np.asarray(weights[0])  # 0 is weights, 1 is biases
        w = w.reshape(w.shape)
        for index in range(model.get_layer(layer_name).output_shape[-1]):
            index_w = np.mean(w[..., index])
            if index_w <= 0:
                continue
            model_layer_weights_dict[(layer_name,index)]=index_w
    # notice!
    model_layer_weights_list = sorted(model_layer_weights_dict.items(), key=lambda x: x[1], reverse=True)

    k = 0
    for (layer_name, index),weight in model_layer_weights_list:
        if k >= top_k:
            break
        model_layer_weights_top_k.append([layer_name,index])
        k += 1


def neuron_selection(model, model_layer_times, model_layer_value, neuron_select_strategy, neuron_to_cover_num, threshold):
    if neuron_select_strategy == 'None':
        return random_strategy(model, model_layer_times, neuron_to_cover_num)

    num_strategy = len([x for x in neuron_select_strategy if x in ['0', '1', '2', '3']])
    neuron_to_cover_num_each = neuron_to_cover_num / num_strategy

    loss_neuron = []
    # initialization for strategies
    if ('0' in list(neuron_select_strategy)) or ('1' in list(neuron_select_strategy)):
        i = 0
        neurons_covered_times = []
        neurons_key_pos = {}
        for (layer_name, index), time in model_layer_times.items():
            neurons_covered_times.append(time)
            neurons_key_pos[i] = (layer_name, index)
            i += 1
        neurons_covered_times = np.asarray(neurons_covered_times)
        times_total = sum(neurons_covered_times)

    # select neurons covered often
    if '0' in list(neuron_select_strategy):
        if times_total == 0:
            return random_strategy(model, model_layer_times, 1)#The beginning of no neurons covered
        neurons_covered_percentage = neurons_covered_times / float(times_total)
        # num_neuron0 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage)
        num_neuron0 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False, p=neurons_covered_percentage)
        for num in num_neuron0:
            layer_name0, index0 = neurons_key_pos[num]
            loss0_neuron = K.mean(model.get_layer(layer_name0).output[..., index0])
            loss_neuron.append(loss0_neuron)

    # select neurons covered rarely
    if '1' in list(neuron_select_strategy):
        if times_total == 0:
            return random_strategy(model, model_layer_times, 1)
        neurons_covered_times_inverse = np.subtract(max(neurons_covered_times), neurons_covered_times)
        neurons_covered_percentage_inverse = neurons_covered_times_inverse / float(sum(neurons_covered_times_inverse))
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)
        num_neuron1 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False,
                                       p=neurons_covered_percentage_inverse)
        for num in num_neuron1:
            layer_name1, index1 = neurons_key_pos[num]
            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
            loss_neuron.append(loss1_neuron)

    # select neurons with largest weights (feature maps with largest filter weights)
    if '2' in list(neuron_select_strategy):
        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]
        k = 0.1
        top_k = k * len(model_layer_times)  # number of neurons to be selected within
        global model_layer_weights_top_k
        if len(model_layer_weights_top_k) == 0:
            neuron_select_high_weight(model, layer_names, top_k)  # Set the value

        num_neuron2 = np.random.choice(range(len(model_layer_weights_top_k)), neuron_to_cover_num_each, replace=False)
        for i in num_neuron2:
            # i = np.random.choice(range(len(model_layer_weights_top_k)))
            layer_name2 = model_layer_weights_top_k[i][0]
            index2 = model_layer_weights_top_k[i][1]
            loss2_neuron = K.mean(model.get_layer(layer_name2).output[..., index2])
            loss_neuron.append(loss2_neuron)

    if '3' in list(neuron_select_strategy):
        above_threshold = []
        below_threshold = []
        above_num = neuron_to_cover_num_each / 2
        below_num = neuron_to_cover_num_each - above_num
        above_i = 0
        below_i = 0
        for (layer_name, index), value in model_layer_value.items():
            if threshold + 0.25 > value > threshold and layer_name != 'fc1' and layer_name != 'fc2' and \
                    layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                    and above_i < above_num:
                above_threshold.append([layer_name, index])
                above_i += 1
                # print(layer_name,index,value)
                # above_threshold_dict[(layer_name, index)]=value
            elif threshold > value > threshold - 0.2 and layer_name != 'fc1' and layer_name != 'fc2' and \
                    layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                    and below_i < below_num:
                below_threshold.append([layer_name, index])
                below_i += 1
        #
        # loss3_neuron_above = 0
        # loss3_neuron_below = 0
        loss_neuron = []
        if len(above_threshold) > 0:
            for above_item in range(len(above_threshold)):
                loss_neuron.append(K.mean(
                    model.get_layer(above_threshold[above_item][0]).output[..., above_threshold[above_item][1]]))

        if len(below_threshold) > 0:
            for below_item in range(len(below_threshold)):
                loss_neuron.append(-K.mean(
                    model.get_layer(below_threshold[below_item][0]).output[..., below_threshold[below_item][1]]))

        # loss_neuron += loss3_neuron_below - loss3_neuron_above

        # for (layer_name, index), value in model_layer_value.items():
        #     if 0.5 > value > 0.25:
        #         above_threshold.append([layer_name, index])
        #     elif 0.25 > value > 0.2:
        #         below_threshold.append([layer_name, index])
        # loss3_neuron_above = 0
        # loss3_neuron_below = 0
        # if len(above_threshold)>0:
        #     above_i = np.random.choice(range(len(above_threshold)))
        #     loss3_neuron_above = K.mean(model.get_layer(above_threshold[above_i][0]).output[..., above_threshold[above_i][1]])
        # if len(below_threshold)>0:
        #     below_i = np.random.choice(range(len(below_threshold)))
        #     loss3_neuron_below = K.mean(model.get_layer(below_threshold[below_i][0]).output[..., below_threshold[below_i][1]])
        # loss_neuron += loss3_neuron_below - loss3_neuron_above
        if loss_neuron == 0:
            return random_strategy(model, model_layer_times, 1)  # The beginning of no neurons covered

    return loss_neuron

def neuron_scale(loss_neuron):
    loss_neuron_new = []
    loss_sum = K.sum(loss_neuron)
    for loss_each in loss_neuron:
        loss_each /= loss_sum
        loss_neuron_new.append(loss_each)
    return loss_neuron_new

def neuron_scale_maxmin(loss_neuron):
    max_loss = K.max(loss_neuron)
    min_loss = K.min(loss_neuron)
    base = max_loss - min_loss
    loss_neuron_new = []
    for loss_each in loss_neuron:
        loss_each_new = (loss_each - min_loss) / base
        loss_neuron_new.append(loss_each_new)
    return loss_neuron_new

def neuron_covered(model_layer_times):
    covered_neurons = len([v for v in model_layer_times.values() if v > 0])
    total_neurons = len(model_layer_times)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_times, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # xrange(scaled.shape[-1])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold: #and model_layer_dict[(layer_names[i], num_neuron)] == 0:
                model_layer_times[(layer_names[i], num_neuron)] += 1

    return intermediate_layer_outputs

def update_coverage_value(input_data, model, model_layer_value):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # xrange(scaled.shape[-1])
        for num_neuron in xrange(scaled.shape[-1]):
            model_layer_value[(layer_names[i], num_neuron)] = np.mean(scaled[..., num_neuron])

    return intermediate_layer_outputs

'''
def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # xrange(scaled.shape[-1])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True

    return intermediate_layer_outputs
'''

def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False


def get_signature():
    now = datetime.now()
    past = datetime(2015, 6, 6, 0, 0, 0, 0)
    timespan = now - past
    time_sig = int(timespan.total_seconds() * 1000)

    return str(time_sig)
