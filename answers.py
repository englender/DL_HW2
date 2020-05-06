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

Explain the effect of depth on the accuracy. What depth produces the best results and why do you think that's the case?
Were there values of L for which the network wasn't trainable? what causes this? Suggest two things which may be done to resolve it at least partially.


**Your answer:
1. We can see in our results that for 32 filters (K=32) we produce the best accuracy when the depth is 2, and the worst
accuracy when the depth is 16. Similarly for 64 filters (K=64) we produce the best accuracy when the depth is 4, and
the worst accuracy is when the depth is 16. Meaning for both filters we get better results from the lower depths rather
than the higher depths. As we saw in the tutorial, adding depth can cause problems just like we experienced in our 
results. A large depth can cause the gradient to vanish or explode on the back propagation by the time it reaches layers 
close to the model input.

2. Originally we ran the experiment and we found that for K=64, L=16 the network wasn't trainable. We suspected this was
happening because the progression towards the gradient direction was too big, and always passed the minimum point. This
assumption was verified when we decreased the learn rate by half and the loss started to decrease as well as the 
accuracy improved. That being said, we still produced lesser results with L=16.



**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""

Analyze your results from experiment 1.2. In particular, compare to the results of experiment 1.1.

**Your answer:
In experiment 1.2 for L=2, we can see that the best results are produced for K=128, followed closely by K=64,256 which
are very similar, and the lowest accuracy is when K=32. In experiment 1.1 we ran the same settings for K=32,64 and 
as expected the accuracy percentage are the same in both experiments for these parameters. Following these results we 
conclude that for L=2 the best number of filters is K=128, a setting that wasn't checked in experiment 1.1

For L=4, we produce similar results as L=2, but we achieve a higher accuracy. The best results are produced for 
K=128,256, followed by K=64 and the lowest accuracy is when K=32. Like with L=2, the accuracy percentage are the same 
as in experiment 1.1 for K=32,64. 

For L=8, we produce the best results for K=258, and as the K drops so does the accuracy.

From experiments 1.1,1.2 we can conclude that there is a linear relation between the K and L, so as the L increases we
produce better results for an increasing K.**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""

Analyze your results from experiment 1.3

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
