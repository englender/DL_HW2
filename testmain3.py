import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
import hw2.cnn as cnn

import hw2.experiments as experiments
from hw2.experiments import load_experiment
from cs236781.plot import plot_fit

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

torch.manual_seed(seed)

net = cnn.ConvClassifier((3,100,100), 10, channels=[32]*4, pool_every=2, hidden_dims=[100]*2)
print(net)

test_image = torch.randint(low=0, high=256, size=(3, 100, 100), dtype=torch.float).unsqueeze(0)
test_out = net(test_image)
print('out =', test_out)

expected_out = torch.load('tests/assets/expected_conv_out.pt')
test.assertLess(torch.norm(test_out - expected_out).item(), 1e-3)


# Test experiment1 implementation on a few data samples and with a small model
experiments.run_experiment(
    'test_run', seed=seed, bs_train=128, batches=100, epochs=100, early_stopping=3,
    filters_per_layer=[64, 128, 256], layers_per_block=8, pool_every=9, hidden_dims=[100],
    model_type='resnet',
)

# There should now be a file 'test_run.json' in your `results/` folder.
# We can use it to load the results of the experiment.
cfg, fit_res = load_experiment('results/test_run_L1_K32-64.json')
_, _ = plot_fit(fit_res)

# And `cfg` contains the exact parameters to reproduce it
print('experiment config: ', cfg)