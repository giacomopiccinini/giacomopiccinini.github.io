# How To Weigh Your Loss

## Introduction

I have recently argued in a separate [blog post](https://giacomopiccinini.github.io/2024/10/21/imbalanced-datasets-should-stay-imbalanced.html) that, in the case of a binary classification problem with high class imbalance, under/over-sampling is a bad practice for it leads to inflated metrics. 

Be that as it may, the actual reason for manipulating the dataset is well-founded: training a classification model in a highly imbalanced case is hard! The choice a vanilla cross-entropy as loss function will most likely result in the model (almost) constantly predicting the majority class, which is useless. 

Assuming you won't touch the dataset for the reason above, the simplest thing you could do is weighing your classes in the loss function. As we’ll see in a moment, there is a *natural choice* for the weights, but it may not fully suit your case, although it provides a solid starting point. Interestingly, the choice of weights entails a trade-off between precision and recall and we'll see how.

## Natural Weights

Let's start by framing the problem. Suppose we have a binary classification problem on a generic, possibly imbalanced, dataset $D$. The dataset $D$ will be composed of two subsets $\mathcal{C}_0$ and $\mathcal{C}_1$, $D = \mathcal{C}_0 \cup \mathcal{C}_1$, where $\mathcal{C}_i$ contains all the elements with label $i$. We shall assume that $\mathcal{C}_0$ is the majority class, i.e.

 $\lvert\mathcal{C}_0\rvert > \lvert\mathcal{C}_1\rvert$ where $\lvert\bullet\rvert$ denotes the cardinality.

In a highly imbalanced case, $|\mathcal{C}_0| \gg |\mathcal{C}_1|$, the reason for the model trivially predicting 0 is that not enough emphasis is put on predicting the minority class: in other words, the model does not have any incentive to get the 1's right and will get better (i.e. reduce its loss) by simply making sure to get the 0's right, which is easy since they are the majority. 

Oversimplifying, and with some abuse of notation, if we let $\lambda$ indicate the order of magnitude for the contribution to the total loss $L$ of a single sample $x$ with ground truth $y$, $\lambda \sim \mathcal{L}(x, y)$ ($\mathcal{L}$ being the loss function), then we'd roughly have

$$
L \sim (|\mathcal{C}_0| + |\mathcal{C}_1|) \lambda \sim |\mathcal{C}_0| \lambda \, .
$$

One can then hope to fix this by introducing some weights that will bring the contributions of $\mathcal{C}_0$ and $\mathcal{C}_1$ to the same order of magnitude so that $\mathcal{C}_0$ doesn't overshadow $\mathcal{C}_1$ anymore. There is a *canonical* or *natural* choice: just give $\mathcal{C}_0$ weight $|\mathcal{C}_1|$ and $\mathcal{C}_1$ weight $|\mathcal{C}_0|$, so that $L \sim |\mathcal{C}_0||\mathcal{C}_1| \lambda$. 

In other words, the **natural weights** $w_i$ for $\mathcal{C}_i$ are given by

$$
(w_0, w_1) = (|\mathcal{C}_1|, |\mathcal{C}_0|)
$$

In fact, we only have **one degree of freedom** which is the ratio $w_r = w_1/w_0$. The reason for this is that the other apparent degree of freedom is "pure gauge" as it is just an overall rescaling of the loss. 

So, we have come to the conclusion that $w_r= |\mathcal{C}_0| / |\mathcal{C}_1|$ is our natural choice. This can (and in fact will) bring us quite far but it's not necessarily the end of the story.

## Exploring The Weights Landscape

The reason why natural weights may not provide a complete picture is that, despite the two contributions to the loss function being of similar magnitude, we cannot assume that precision and recall will also align to our needs. 

**Caution here**: precision and recall need not be equal as they carry different *business* values. In some scenarios, higher precision may be more valuable than higher recall, while in others, the reverse may hold true. This relationship is not universal and should be evaluated on a case-by-case basis. Thus, the default choice of weights may not adequately represent our priorities regarding the trade-off between precision and recall. 

Hence the question: *how to tune $w_r$ to better align with my requirements?*

A mathematically sound proof is not straightforward (I think), but I will provide some heuristic and some computational results supporting it.

When setting $w_r$ to a low value, hence not penalising or even emphasising the *majority* class, the model tends to predict 0 almost exclusively. This means that it will **fail to identify** the vast majority of actual class 1 instances. Put differently, we'll be flooded with *false negatives*. Conversely, the model will only predict class 1 when it is highly confident that a sample truly belongs to that class. This means we can anticipate many *false negatives* alongside very few *false positives*, resulting in high precision but low recall.

Conversely, when $w_r$ is set to a high value, the model prioritises the minority class, class 1. For sufficiently high $w_r$ we end up in a situation opposite to the one we started from: even if $|\mathcal{C}_0| \gg |\mathcal{C}_1|$ still holds true, there is so much emphasis on getting $\mathcal{C}_1$ right that the model finds more convenient to simply blindly predicting 1. As a result, we will end up with a lot of *false positives* and few *false negatives*, or high recall but low precision.

Given these two extreme cases, we expect the model to interpolate between them when varying $w_r$. One might anticipate precision and recall to be roughly equal around the *natural weights*. In my experience this is somewhat true, but adjustments are most often needed. 

We are now in the position to run some experiments. I have used a fairly large dataset with 88:12 imbalance; unfortunately it's a private one so I am not in the position to share more (but there wasn't any particular feature worth highlighting). Given this imbalance, the natural weight ratio is $w_r = 88/12 \sim 7$. I capped the epochs at 1 (since the dataset was already large and we were aiming for a fine-tuning) and estimated how the recall and precision varied across the various choices. I ended up with this chart (notice that the $x$-axis is in log scale!)


![Chart](/assets/images/precision-vs-recall.png)

We can see that the extremes are indeed as we expected them to be. Also, precision and recall do "match" at some higher value than the natural $w_r$ (or here "expected best ratio", with $\log 7 \sim 2$). 

I find having this chart in mind when tuning the weights super-useful! Of course, there's unfortunately still a lot of alchemy involved in getting your model right, but it's a good starting point.