# Imbalanced Datasets Should Stay Imbalanced

## Introduction

The common lore in Machine Learning books is that, in the unlucky case you are presented with an imbalanced dataset for a classification task, you should re-balance it right away via under- or over-sampling. 

Whilst this recipe has gone down in the textbooks, it is *borderline criminal*. First, datasets “in nature” are never balanced: a realistic dataset for binary classification task will be typically imbalanced with a ratio of 99:1 for the majority vs minority class. Second, artificially modifying a dataset is never a good idea and I am going to explain why in a second. 

To do this, I will start from some maths and leave the code at the very end, just to test out and verify our findings. 

## A mathematical explanation

Let’s set the stage. Suppose we have a binary classification problem on a generic, possibly imbalanced, dataset $D$. The dataset $D$ will be composed of two subsets $\mathcal{C}_0$ and $\mathcal{C}_1$, $D = \mathcal{C}_0 \cup \mathcal{C}_1$, where $\mathcal{C}_i$ contains all the elements with label $i$. The cardinalities of these subsets, i.e. the number of elements they contain, are indicated with

$$
Z = |\mathcal{C}_0|\, , \qquad O = |\mathcal{C}_1| \, . 
$$

Assuming the minority class to always be that with label 1, it is useful to introduce a quantity $\rho$ defined by

$$
\rho = \frac{Z}{O}
$$

In case of perfectly balanced dataset, $\rho=1$. For an imbalanced dataset, $\rho > 1$. 

In the following we shall assume $D$ to represent some imbalanced dataset and we will indicate with $D_B$ its balanced version obtained by downsampling the majority class. That is, we take $D$ and remove elements to $\mathcal{C}_0$ until its cardinality matches that of $\mathcal{C}_1$ (or, equivalently, $\rho_B = 1$). 

Now, before training a Machine Learning model, we should also split $D$ into a train and test partition, $D = D^{TR} \cup D^{TE}$ (the validation part is re-absorbed into $D^{TR}$). If, and that's a big *if*, we were to somehow re-balance part of the dataset we would have four possibilities to choose from:
1. Imbalanced train set + Imbalanced test set
2. Imbalanced train set + Balanced test set
3. Balanced train set + Balanced test set
4. Balanced train set + Imbalanced test set

Option 1. is the *natural* option,  where the dataset has not been touched. We shall argue later on that this is the best option. 

Option 2. is certainly the worst choice: we get the headaches of an imbalanced train set combined with the unreliability of a perfectly balanced test set. 

Option 3. is the one often appearing in textbooks. We shall see how this can lead to misleading results. 

Option 4. is arguably the second best choice, albeit not really warmly recommended. 

If you have followed the discussion so far, you should have noticed that what makes one of the options above not suitable is the presence of a balanced test set. The thing is, a model is *better* only if it performs *better* on unseen data that reflect the **real** scenario the model will face once deployed. Here the keyword is "real": if the dataset is natively imbalanced, there is no way a balanced test set is "real": it will necessarily be an oversimplification of a much more complex case. This simplification leads to inflated performance on the balanced test set and thus to inevitable disappointment once the system "goes live". 

 Let me now explain in more detail what I mean by “inflated performance”. 
 
### Inflated performance

Let us assume that some model has been trained on the train set $D^{TR}$ . The balance of $D^{TR}$ is irrelevant for the subsequent discussion and we will simply neglect it. Instead, we shall completely focus on the test set $D^{TE}$ and indicate it with $D$ for brevity.  

To define the concept of performance, we need to pick a metric first. There’s a plethora of them but we'll try to keep it simple and pick the most common in these cases, i.e. the $F_1$ metric defined as

$$
F_1 = 2 \frac{p \cdot r}{p + r}
$$

where $p$ and $r$ indicate *precision* and *recall*, respectively. In turn, these quantities are defined in terms of the *confusion matrix* elements ($TP$, $TN$, $FP$, $FN$) as

$$
p = \frac{TP}{TP + FP}\, , \qquad r = \frac{TP}{TP + FN}\, .
$$

Let’s explore these a little bit further. The $TN$ do no enter the game at all. However, we can rephrase the $FP$ as 

$$
FP = Z \cdot \mathrm{fpr} = \rho \cdot O \cdot \mathrm{fpr}\, , \qquad \mathrm{fpr} = \frac{FP}{Z} = \frac{FP}{TN+FP}
$$

where $\mathrm{fpr}$ indicates the *false positive ratio*. The reason for introducing the latter is that, *in principle*, for a well-trained model its value should not depend on the dataset size, nor on the dataset balance: if the model generalises well, this ratio should remain approximately constant irrespective of the dataset being composed of 100 or 1 million elements, and is not sensitive to the true number of elements with label 1. On the contrary, $FP$ is obviously very much dependent on dataset size. 

To start out, let’s assume we have a balanced dataset $D_B$ composed of $N_B$ elements evenly split between 0’s and 1’s (i.e. $\rho_B = 1$, $Z_B = O_B = N_B/2$).  Then,

$$
p_B= \frac{TP_B}{TP_B + \rho_B \, O_B \, \mathrm{fpr}} \, , \qquad r_B = \frac{TP_B}{TP_B + FN_B} \, ,
$$

where $TP_B$ (and similar) indicate the confusion matrix element for the balanced case. The reason why we didn’t bother rewriting the recall in a similar fashion will be clear in a second. 

Now, suppose we realise this dataset is not realistic and hence not well suited to test the model performance due to its $\rho_B$ artificially set to 1. We would then go and extend the dataset by adding back in all the 0’s we discarded in order to reach that ratio. Here I am implicitly assuming we re-balanced the dataset via downsampling, for two reasons: first, it’s the easiest option to achieve that result; second, upsampling requires, in practice, the creation of synthetic data and, if we had to cure a typical 99:1 disparity, we would need **a lot** of them making their generation an extremely delicate procedure.

So, if we are just (re)adding 0’s until we recover the original imbalanced dataset $D$, we are not touching the 1’s, meaning that $O_B = O$. For the same reason, the number of false negatives and true positives will not change, hence $FN_B = FN$ and $TP_B = TP$. As a consequence, recall will not change either $r_B = r$. This is why we didn't bother rewrite it. However, the precision will change because the number of $FP$ will increase, 

$$
p = \frac{TP}{TP + \rho \, O \, \mathrm{fpr}} = p_B \frac{TP + O \, \mathrm{fpr}}{TP + \rho \, O \, \mathrm{fpr}} = p_B \frac{r + \mathrm{fpr}}{r + \rho \, \mathrm{fpr}} = p_B \frac{1 + \mathrm{fpr}/r}{ 1 + \rho \, \, \mathrm{fpr}/r}
$$ 

where in the penultimate step we have used a rewriting of the definition of recall $r = TP/O$. 

Notice how, in the last formula, everything aside from $\rho$ on the right-hand-side depends on the model performance on the balanced dataset: we can predict how the precision (and hence the $F_1$) will degrade as we move the dataset back to its original balance! Moreover, the cardinality of the dataset has dropped out, meaning that any analysis and conclusion we’ll draw will not depend on dataset size, i.e. any pathological behaviour will not be cured by simply making the dataset larger (if preserving the balance, of course). Finally, notice how the presence of $\rho$ at the denominator implies that the precision on the unbalanced dataset can only decrease, no matter what. 

In a somewhat abstract way, this is why we can't test on a balanced dataset: we will end up having a lot more false positives in production than we expected! 

There's more to the story though, but this is best covered by making things slightly more concrete. 

### A little more concrete

Suppose we have a balanced dataset $D_B$ . It makes sense to assume that, whatever their precise values, $TP = TN$ and $FP = FN$ for a well-trained model. This is by no means mandatory, as $FP$ and $FN$ carry in general different **business** value, and one might favour one over the other in a realistic scenario; however one typically hopes to find a good balance between these two. 

This choice implies that precision and recall should be equal $p_B = r$. Finally, using the definition we note that $\mathrm{fpr}:=1 -TN_B/Z_B = 1-TP/O = 1-r=1-p_B$. As a consequence we can simplify things down to 

$$
p(\rho) = p_B \frac{1/r}{1 + \rho(1/r - 1)} = p_B\frac{1}{r + \rho(1-r)} = p_B \frac{1}{p_B + \rho(1-p_B)}
$$

Whilst this is all good, plotting how the precision $p(\rho)$ varies with $\rho$ for a fixed value of $p_B$ is extremely telling. 
![[/giacomopiccinini.github.io/assets/images/precision-fixed-pb.png]]
What we see here is that, unless we start from an extremely high (and hence unrealistic) value for $p_B$ , the precision is going to drop extremely quickly as the dataset is brought back to its natural imbalance. And, in particular, the lower $p_B$ the steepest the downfall. 

Therefore, unless you trained a super-good model with 99.9+% precision on the balanced test set, its value on the realistic test set will drop a lot and very quickly, making what you once thought a good model a completely unusable piece of software.

