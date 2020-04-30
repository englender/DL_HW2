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