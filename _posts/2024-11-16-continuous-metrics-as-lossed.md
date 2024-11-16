# Continuous Metrics as Losses

## Introduction

In classification problems, it is common practice to adopt two different standards for evaluating a model's performance.

During training, a loss function like Cross Entropy is used. While ubiquitous, its value at any given point is not particularly informative or easily interpretable. For example:

1. **Cross Entropy is unbounded** (from above), making it difficult to assess model performance intuitively. We are accustomed to well-defined scales with clear maximum and minimum values (e.g., a 1-to-10 rating scale).
2. **Loss function values depend on weights**: Cross Entropy often incorporates non-trivial weights to address class imbalances in the dataset. Different weight configurations lead to different loss values, making comparisons across projects—or even within experiments—challenging.

In contrast, metrics with bounded, interpretable ranges—such as $F_\beta$-scores, precision, and recall—are widely understood. For example, saying *"My model has a 90% $F_1$ score"* communicates good performance without additional context. In comparison, *"My model has a Cross Entropy of 2.53"* provides little intuitive meaning.

At a higher level, these metrics can be seen as **business metrics**, directly reflecting how a model's performance compares to competitors, which may influence commercial success.

However, this creates a tension: we train models by minimizing a loss function (e.g., Cross Entropy) that is not directly used to evaluate their ultimate performance.

The technical reason for this discrepancy is that backpropagation requires the loss function to be differentiable, whereas metrics like $F_1$-score are not. Metrics typically involve thresholding operations: the softmaxed logits output by the model are discretized into 0 or 1 based on a threshold  $t$ (commonly  $t = 0.5$).

To address this, one approach is to design a **differentiable approximation** of these metrics. This involves defining what constitutes a *continuous* version of a False Positive (FP).

## Entering the continuous

Let's take a binary classification problem, where we have a vector of ground truths $y$ and a set of softmaxed predictions $p$. When needed, we will indicate the $i$-th element of $y$ and $p$ (and any other vector, in fact) with an index, e.g. $p_i$ with $i=1, \dots, B$ where $B$ is the batch size. Notice that $y \in \{0, 1 \}^B$ while $p \in [0, 1]^B$.

We can use $y$ to define two projection operators, $ \pi_0 $ and $ \pi_1 $, that isolate the negative and positive classes, respectively. These operators modify $ p $ by zeroing out all entries that do not correspond to the negative class ($ \pi_0 $) or positive class ($ \pi_1 $) in the ground truth vector $ y $. For example:

$$
p = [0.1, 0.6, 0.3], \quad y = [0, 1, 1], \quad \pi_0(p) =[0.1, 0, 0], \quad \pi_1(p) = [0, 0.6, 0.1] \, .
$$

The projectors act via the Hadamard product $\odot$ and are trivially defined by 

$$
\pi_0 = (1-y)\, \qquad \pi_1 = y \, .
$$

Using these projections, we can now compute the **expected** negative ($ N $) and positive ($ P $) predictions from $ p $:

$$
N = \pi_0(p), \qquad P = \pi_1(p) \, .
$$

Next, we need to define what constitutes *True* or *False* predictions in a differentiable context. As mentioned earlier, we cannot use a threshold-based approach, as it introduces the discretization we aim to avoid. Here are the definitions we will adopt, with explanations to follow:

$$
TP = \sum_i P_i, \qquad FP = \sum_i N_i, \qquad TN = \sum_i (\pi_0 - N)_i, \qquad FN = \sum_i (\pi_1 - P)_i \, .
$$

The reasoning behind these definitions is as follows: Consider a case where the model predicts $ p_i = 0.8 $ for some $ i $. If the corresponding ground truth is $ y_i = 1 $, this prediction is 0.8 (or 80%) correct, and we interpret it as contributing 80% to the True Positives (TP). Conversely, if $ y_i = 0 $, the prediction is 80% incorrect, contributing 80% to the False Positives (FP), assuming a threshold of $ t = 0.5 $ was applied.

We can express these calculations more explicitly as:

$$
TP = y \cdot p, \qquad FP = (1-y) \cdot p, \qquad TN = (1-y) \cdot (1-p), \qquad FN = y \cdot (1-p) \, .
$$

Here, the symbol $ \cdot $ represents the element-wise scalar product. Using these components, we can construct any desired metric—such as $ F_\beta $-score, precision, recall, or accuracy—and use it as a differentiable loss function for the classifier.

## PyTorch Implementation

Here's finally a PyTorch implementation of losses built with this technique out of a bunch of metrics. Ready to be imported and used!

```python
import torch


def get_continuous_confusion_matrix(p: torch.Tensor, y: torch.Tensor):
    """Get the continuous confusion matrix
    p: (B, )
    y: (B, )
    """

    TP = torch.dot(p, y)
    FP = torch.dot(p, 1 - y)
    FN = torch.dot(1 - p, y)
    TN = torch.dot(1 - p, 1 - y)

    return TP, TN, FP, FN


class AccuracyLoss(torch.nn.Module):
    """Class implementation of the continuous Accuracy loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the accuracy loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the loss
        return -((TP + TN) / (TP + TN + FP + FN))


class PrecisionLoss(torch.nn.Module):
    """Class implementation of the continuous Precision loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the precision loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, _TN, FP, _FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the precision loss
        return -(TP / (TP + FP))


class RecallLoss(torch.nn.Module):
    """Class implementation of the continuous Recall loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the recall loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, _TN, _FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the recall loss
        return -(TP / (TP + FN))


class FBetaLoss(torch.nn.Module):
    """Class implementation of the continuous F-beta loss."""

    def __init__(self, beta: float = 1.0):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

        # Store beta
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the F-beta loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, _TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute precision and recall
        P = TP / (TP + FP)
        R = TP / (TP + FN)

        # Compute the F-beta loss
        return -((1 + self.beta**2) * P * R / ((self.beta**2 * P) + R))


class MarkednessLoss(torch.nn.Module):
    """Class implementation of the continuous Markedness loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the F-beta loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the F-beta loss
        return -(TP / (TP + FP) - FN / (TN + FN))


class InformednessLoss(torch.nn.Module):
    """Class implementation of the continuous Informedness loss."""

    def __init__(self):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the F-beta loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute the F-beta loss
        return -(TP / (TP + FN) - FP / (TN + FP))


class PhiBetaLoss(torch.nn.Module):
    """Class implementation of the continuous phi-beta loss."""

    def __init__(self, beta: float = 1.0):
        # Use constructor of parent class
        super().__init__()

        # Sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

        # Store beta
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Apply the F-beta loss"""

        # Convert to probs
        probabilities = self.sigmoid(logits)

        # Reshape to (B, ) and compute the continuous confusion matrix
        TP, TN, FP, FN = get_continuous_confusion_matrix(
            p=probabilities.view(-1), y=targets.view(-1)
        )

        # Compute informedness and markedness
        i = TP / (TP + FN) - FP / (TN + FP)
        m = TP / (TP + FP) - FN / (TN + FN)

        return (1 + self.beta**2) * i * m / (self.beta**2 * m + i)
```