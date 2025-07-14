# MPDC-net
Multi-Perspective Dynamic Consistency Network

Ours work has been accepted by Scientific reports，DOI：https://doi.org/10.1038/s41598-025-03124-2

Title：Multi-perspective dynamic consistency learning for semisupervised medical image segmentation

Abstract：Semi-supervised learning (SSL) is an effective method for medical image segmentation as it alleviates the dependence on clinical pixel-level annotations. Among the SSL methods, pseudo-labels and consistency regularization play a key role as the dominant paradigm. However, current consistency regularization methods based on shared encoder structures are prone to trap the model in cognitive bias, which impairs the segmentation performance. Furthermore, traditional fixed-threshold-based pseudo-label selection methods lack the utilization of low-confidence pixels, making the model’s initial segmentation capability insufficient, especially for confusing regions. To this end, we propose a multi-perspective dynamic consistency (MPDC) framework to mitigate model cognitive bias and to fully utilize the low-confidence pixels. Specially, we propose a novel multi-perspective collaborative learning strategy that encourages the sub-branch networks to learn discriminative features from multiple perspectives, thus avoiding the problem of model cognitive bias and enhancing boundary perception. In addition, we further employ a dynamic decoupling consistency scheme to fully utilize low-confidence pixels. By dynamically adjusting the threshold, more pseudo-labels are involved in the early stages of training. Extensive experiments on several challenging medical image segmentation datasets show that our method achieves state-of-the-art performance, especially on boundaries, with significant improvements.

Experiment Detail：

The "data" file contains the train, test, and validation splits for the ACDC, PROMISE12, and Polyp datasets. Polyp datasets includes CVC-ClinicDB and Kvasi

In the "model" file, we provide our pre-trained model, as well as the testing details and results on the test dataset.
