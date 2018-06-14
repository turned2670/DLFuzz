# Demo

## Dependencies

Part of the needed dependencies are listed as below for the experiment
```
cuda 8.0
cudnn 5.1

Python 2.7.14
tensorflow-gpu (1.2.1)
Keras (2.1.3)
h5py (2.7.1)
Pillow (5.0.0)
opencv-python
```

To install on Linux(ubuntu)
```
When installing tensorflow 1.2.1, you can download the specified version on PYPI or the website of tensorflow.
pip install tensorflow-gpu
pip install keras
pip install Pillow
pip install h5py
pip install opencv-python
```

## To run

generate adversarial examples for ImageNet
```
cd ImageNet
python gen_diff.py [2] 0.25 10 0602 3 vgg16
#meanings of arguments
#python gen_diff.py 
[2] -> the list of neuron selection strategies
0.25 -> the activation threshold of a neuron
10 -> the number of neurons selected to cover
0602 -> the folder holding the adversarial examples generated
3 -> the number of times for mutation on each seed
vgg16 -> the DL model under test
```

generate adversarial examples for MNIST
```
cd MNIST
python gen_diff.py [2] 0.5 5 0602 5 model1
#meanings of arguments are the same as above
```