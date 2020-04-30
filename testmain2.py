import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
import hw2.optimizers as optimizers
import hw2.training as training
import hw2.blocks as blocks
import hw2.answers as answers
from torch.utils.data import DataLoader
from cs236781.plot import plot_fit
from hw2.grad_compare import compare_block_to_torch

seed = 42
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()


# Test VanillaSGD
torch.manual_seed(42)
p = torch.randn(500, 10)
dp = torch.randn(*p.shape)*2
params = [(p, dp)]

vsgd = optimizers.VanillaSGD(params, learn_rate=0.5, reg=0.1)
vsgd.step()

expected_p = torch.load('tests/assets/expected_vsgd.pt')
diff = torch.norm(p-expected_p).item()
print(f'diff={diff}')
test.assertLess(diff, 1e-3)

data_dir = os.path.expanduser('~/.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

# print(f'Train: {len(ds_train)} samples')
# print(f'Test: {len(ds_test)} samples')


# Overfit to a very small dataset of 20 samples
batch_size = 10
max_batches = 2
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

# Get hyperparameters
hp = answers.part2_overfit_hp()

torch.manual_seed(seed)

# Build a model and loss using our custom MLP and CE implementations
model = blocks.MLP(3 * 32 * 32, num_classes=10, hidden_features=[128] * 3, wstd=hp['wstd'])
loss_fn = blocks.CrossEntropyLoss()

# Use our custom optimizer
optimizer = optimizers.VanillaSGD(model.params(), learn_rate=hp['lr'], reg=hp['reg'])

# Run training over small dataset multiple times
# trainer = training.BlocksTrainer(model, loss_fn, optimizer)
# best_acc = 0
# for i in range(20):
#     res = trainer.train_epoch(dl_train, max_batches=max_batches)
#     best_acc = res.accuracy if res.accuracy > best_acc else best_acc
#
# test.assertGreaterEqual(best_acc, 98)

# Define a larger part of the CIFAR-10 dataset (still not the whole thing)
batch_size = 50
max_batches = 100
in_features = 3*32*32
num_classes = 10
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size//2, shuffle=False)


# Define a function to train a model with our Trainer and various optimizers
def train_with_optimizer(opt_name, opt_class, fig):
    torch.manual_seed(seed)

    # Get hyperparameters
    hp = answers.part2_optim_hp()
    hidden_features = [128] * 5
    num_epochs = 10

    # Create model, loss and optimizer instances
    model = blocks.MLP(in_features, num_classes, hidden_features, wstd=hp['wstd'])
    loss_fn = blocks.CrossEntropyLoss()
    optimizer = opt_class(model.params(), learn_rate=hp[f'lr_{opt_name}'], reg=hp['reg'])

    # Train with the Trainer
    trainer = training.BlocksTrainer(model, loss_fn, optimizer)
    fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches)

    # fig, axes = plot_fit(fit_res, fig=fig, legend=opt_name)
    # return fig

fig_optim = None
# fig_optim = train_with_optimizer('vanilla', optimizers.VanillaSGD, fig_optim)

# fig_optim = train_with_optimizer('momentum', optimizers.MomentumSGD, fig_optim)
# fig_optim

# fig_optimim = train_with_optimizer('rmsprop', optimizers.RMSProp, fig_optim)
# fig_optim


# Check architecture of MLP with dropout layers
mlp_dropout = blocks.MLP(in_features, num_classes, [50]*3, dropout=0.6)
print(mlp_dropout)
test.assertEqual(len(mlp_dropout.sequence), 10)
for b1, b2 in zip(mlp_dropout.sequence, mlp_dropout.sequence[1:]):
    if str(b1) == 'relu':
        test.assertTrue(str(b2).startswith('Dropout'))
test.assertTrue(str(mlp_dropout.sequence[-1]).startswith('Linear'))

# Test end-to-end gradient in train and test modes.
print('Dropout, train mode')
mlp_dropout.train(True)
for diff in compare_block_to_torch(mlp_dropout, torch.randn(500, in_features)):
    test.assertLess(diff, 1e-3)

print('Dropout, test mode')
mlp_dropout.train(False)
for diff in compare_block_to_torch(mlp_dropout, torch.randn(500, in_features)):
    test.assertLess(diff, 1e-3)