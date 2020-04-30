r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr = 0.05
    reg = 0.05
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.01 #0.01
    lr_vanilla =  0.02 #0.05
    lr_momentum = 0.003 #0.01
    lr_rmsprop = 0.0003
    reg = 0.005
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.005
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer: 
1. The graphs that were computed for the three dropout configurations match our expectations. We can see in the graphs
that the model with no dropout clearly overfits on the train data (ie. the train loss is very low but the test loss 
is high), as expected. On the other hand, when adding a dropout we can see that although the train loss with dropout is 
higher than the train loss with no dropout, the test loss acts differently, and the test loss with dropout is lower 
than the test loss without dropout. This observation indicates an improvement of the generalization of the model when 
dropout is added, which correlates with the goal of the dropout.
2. We can see in the graphs computed for the low-dropout setting and high-dropout setting that the low-dropout setting
achieves better accuracy with the test-set. This is due to underfitting of the model with high-dropout, the high-dropout 
model drops too many neurons which can reduce the efficiency of the training process. We can learn from our results that
using dropout can improve the overfit of the model, however too big of a dropout can cause the opposite reaction 
(underfit).      
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer: When training a model with the cross-entropy loss function, it is possible for the test loss to increase
for a few epochs while the test accuracy increases also.
This can be explained by the following scenario: examples with a small margin error that are predicted wrong can be
changed slightly and make the prediction correct, thus the accuracy increases. Simultaneously, two other things can 
happen that increase the loss while the accuracy increases as explained above. Examples with a big margin error that are
predicted wrong can increase their margin error thus the loss increases. Also, examples with very good predictions (by
a big margin) can get a little worse but still be predicted correctly, also affecting the loss and increasing it. When 
all these events happen we can see the above scenario happen.   
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
