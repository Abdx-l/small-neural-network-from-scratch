# Small Neural Network from Scratch

A NumPy implementation of dense layers, ReLU, softmax cross-entropy, backpropagation, mini-batch SGD, momentum, and learning-rate decay. It can train on MNIST IDX files supplied locally and includes a small synthetic demo.

```sh
python train.py --demo --epochs 100
python train.py --images train-images-idx3-ubyte --labels train-labels-idx1-ubyte --epochs 5
python -m unittest discover -s tests -v
```

The training script prints epoch loss and accuracy. No deep-learning framework is used; only NumPy is required. The tests compare analytic gradients with finite differences and check learning on a small classification problem.

