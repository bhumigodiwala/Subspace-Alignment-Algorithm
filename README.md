# Subspace Alignment Algorithm for Domain Adaptation

Subspace Alignment Algorithm is a new Domain Adaptation algorithm that uses a domain invariant feature space. In this algorithm, the source and the target domains are represented using Eigenvectors. It generates a feature space that allows mapping of the source dataset to the target dataset so that the source data closely aligns with the target data.

<img width="968" alt="Screenshot 2023-11-14 at 4 32 36 PM" src="https://github.com/bhumigodiwala/Subspace-Alignment-Algorithm/assets/62346064/95613ec4-5720-4a44-8c04-ec809d20c9f6">

The aim of this Analysis is to verify the efficiency of the Subspace Alignment Algorithm. We are considering the case of Supervised Learning to explore the impacts of applying Subspace Alignment Algorithm. The perfomance metrics considered for comparison with a supervised learning classifier is Accuracy and KL Divergence.

The key observations suggest:
1. The datasets need to be quite large and consist of number of features
2. The Accuracy varies when we randomly sample fractions of source data for different random seed values. However, the accuracy converges once we randomly sample approximately 75-80% of the source data
3. Applying Subspace Alignment and PCA drastically reduces the KL divergence values between the source and the target datasets.
4. There is reduction in KL divergence for data specific to every class

To run and observe the results:

```
python3 evaluation.py
```