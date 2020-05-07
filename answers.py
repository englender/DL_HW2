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
    wstd = 0.01
    lr_vanilla = 0.02
    lr_momentum = 0.003
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
**Your answer:
1. We can see in our results that for 32 filters (K=32) we produce similar results for L=2,4 and for L=8,16 the network 
isn't trainable. When we increase the filters to K=64, we get very similar results as for K=32. So we can conclude that 
for the smaller depths, when the network is trainable, the different depths don't have a big affect on the accuracy. 
For the parameters that we check in this experiment (i.e a low K) the complexity and diversity of the extracted features 
are low. So for large depths the network is not trainable, and for the lower depths because the complexity and diversity 
are low the differences between them are small. 

2. We can see in our results that the network isn't trainable for L=8,16 for both K values that were checked. We assume 
that this is caused by the vanishing gradient problem, because of the big depth of the network. This can be solved by 
using Batch Normalization, which will normalize the gradient in each layer and help prevent it from vanishing. Another 
solution to this problem is to add skip connections (like we will see in the next experiments), which will help the 
gradient to flow back through the layers without vanishing.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:
In experiment 1.2 for L=2, we produce similar results for all the K values. This can be explained by the low depth of 
the network which causes the feature extraction to be simpler and thus decreasing the affect of different K values.
For L=4 we can see that when we increase the depth more complex features can be created and because of that we see 
larger differences between the different K values. Our results show that for L=4 we achieve the est accuracy for K=256.
For L=8, we start seeing that the network isn't trainable for some K values, which can be explained because the network
is deep thus causing the gradiant to vanish as we explained before.  
In conclusion, when using more filters more diverse features can be extracted, and by making the network deeper the 
features can be more complex. As seen in our results for a low L value the network isn't deep enough to extract more 
complex features derived by a large K value.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:
This experiment is the first time we use several different convolutions for every layer (L). Also we can observe that in 
this experiment we achieve our highest test accuracy. For every L value we have several convolutional layers, the first 
one having K=64 and the next layers have increasing K values, enabling us to extract more complex features, and produce
better results. However this structure deepens the network and we see the for our higher L values that we checked 
(L=3,4) the network doesn't learn.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:
In this experiment we use the ResNet model as opposed to the previous experiments where we used the CNN model. 
In the first part of the experiment we use a constant value of K=32 and depths varying from 8,16,32. When compared to 
experiment 1.1 we can see a significant improvement, especially as the L values increase, where as in experiment 1.1 for
the same L values the network wasn't trainable. 
In the second part where we trained with different convulotional layers, we expirienced the same phenomenon. With our 
ResNet model we achieved better results, and where able to train the network with large depths, as opposed to the CNN
model that wasn't trainable with those depths.
These results are consistent with our expectations, the ResNet model enables the gradient to propagate back from deeper
layers and because of that we can build more complex features resulting in better results. 
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:
1. In this part we added a couple of features to our model that we thought would improve some obstacles that we faced 
in the previous experiments. A problem that we experienced and explained before is the vanishing gradient in large 
depths. In order to overcome that we added Batch Normalization and skip connections (as explained in question 1.2).
Because we wanted to train our model on larger depths, we decided to add dropout to avoid overfitting on our training 
set as we learned from the tutorials. We experimented with several dropout values, and decided on a final value of
dropout=0.2, which is smaller than the values used in part 2.

2. As we expected these additions produced better results than in part 1. We achieved better generalization as
well as higher accuracy. In part 1 we saw from both the CNN model and ResNet model some overfitting, that is reflected
by the improvement of the train accuracy and decrease in the test accuracy. Moreover in part 2 we train the model with 
large depths and still produce high accuracy, as opposed to part 1 where the models didn't train with these large depths.
**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
