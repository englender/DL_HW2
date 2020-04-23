import torch
import unittest

test = unittest.TestCase()
import hw2.blocks as blocks

from hw2.grad_compare import compare_block_to_torch
N = 100
in_features = 200
num_classes = 10


def test_block_grad(block: blocks.Block, x, y=None, delta=1e-3):
    diffs = compare_block_to_torch(block, x, y)

    # Assert diff values
    for diff in diffs:
        test.assertLess(diff, delta)


if __name__ == '__main__':
    # Test Linear
    fc = blocks.Linear(in_features, 1000)
    x_test = torch.randn(N, in_features)

    # Test forward pass
    z = fc(x_test)
    test.assertSequenceEqual(z.shape, [N, 1000])

    # Test backward pass
    test_block_grad(fc, x_test)

    # Test ReLU
    relu = blocks.ReLU()
    x_test = torch.randn(N, in_features)

    # Test forward pass
    z = relu(x_test)
    test.assertSequenceEqual(z.shape, x_test.shape)

    # Test backward pass
    test_block_grad(relu, x_test)

    # Test Sigmoid
    sigmoid = blocks.Sigmoid()
    x_test = torch.randn(N, in_features)

    # Test forward pass
    z = sigmoid(x_test)
    test.assertSequenceEqual(z.shape, x_test.shape)

    # Test backward pass
    test_block_grad(sigmoid, x_test)
