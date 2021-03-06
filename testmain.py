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

    # # Test CrossEntropy
    cross_entropy = blocks.CrossEntropyLoss()
    scores = torch.randn(N, num_classes)
    labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)

    # Test forward pass
    loss = cross_entropy(scores, labels)
    expected_loss = torch.nn.functional.cross_entropy(scores, labels)
    test.assertLess(torch.abs(expected_loss - loss).item(), 1e-5)
    print('loss=', loss.item())

    # Test backward pass
    test_block_grad(cross_entropy, scores, y=labels)

    # Test Sequential
    # Let's create a long sequence of blocks and see
    # whether we can compute end-to-end gradients of the whole thing.

    seq = blocks.Sequential(
        blocks.Linear(in_features, 100),
        blocks.Linear(100, 200),
        blocks.Linear(200, 100),
        blocks.ReLU(),
        blocks.Linear(100, 500),
        blocks.Linear(500, 200),
        blocks.ReLU(),
        blocks.Linear(200, 500),
        blocks.ReLU(),
        blocks.Linear(500, 1),
        blocks.Sigmoid(),
    )
    x_test = torch.randn(N, in_features)

    # Test forward pass
    z = seq(x_test)
    test.assertSequenceEqual(z.shape, [N, 1])

    # Test backward pass
    test_block_grad(seq, x_test)

    # Test MLP architecture
    N = 100
    in_features = 10
    num_classes = 10
    for activation in ('relu', 'sigmoid'):
        mlp = blocks.MLP(in_features, num_classes, hidden_features=[100, 50, 100], activation=activation)
        test.assertEqual(len(mlp.sequence), 7)

        num_linear = 0
        for b1, b2 in zip(mlp.sequence, mlp.sequence[1:]):
            if (str(b2).lower() == activation):
                test.assertTrue(str(b1).startswith('Linear'))
                num_linear += 1

        test.assertTrue(str(mlp.sequence[-1]).startswith('Linear'))
        test.assertEqual(num_linear, 3)

        # Test MLP gradients
        # Test forward pass
        x_test = torch.randn(N, in_features)
        labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)
        z = mlp(x_test)
        test.assertSequenceEqual(z.shape, [N, num_classes])

        # Create a sequence of MLPs and CE loss
        seq_mlp = blocks.Sequential(mlp, blocks.CrossEntropyLoss())
        loss = seq_mlp(x_test, y=labels)
        test.assertEqual(loss.dim(), 0)
        print(f'MLP loss={loss}, activation={activation}')

        # Test backward pass
        test_block_grad(seq_mlp, x_test, y=labels)