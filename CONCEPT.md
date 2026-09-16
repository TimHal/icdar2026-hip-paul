# Research Concept: Self-Supervised Class-wise Autoencoders

This document describes the research motivation, theoretical foundation, and current progress of the class-wise autoencoder framework for dataset analysis and characterization.

## Research Motivation

### The Problem

Modern machine learning depends on large labeled datasets, but:
- Labeling is expensive and time-consuming
- Real-world datasets often contain noisy or incorrect labels
- Understanding dataset difficulty is critical for resource allocation
- No universal metric exists for dataset characterization across domains

### The Opportunity

What if we could:
1. Estimate dataset difficulty without extensive training?
2. Detect mislabeled samples automatically?
3. Cluster and classify with minimal or no labels?
4. Quantify sample-level difficulty with a single metric?

This is where Reconstruction Error Ratios (RER) come in.

## The RER Framework

### Core Idea

Train K separate autoencoders, one for each class. For a sample x with label c, compute:

```
χ(x^c) = Δ^c(x^c) / min_{c' ≠ c} Δ^{c'}(x^c)
```

Where:
- `Δ^c(x)` is the reconstruction error of sample x using class c's autoencoder
- `χ(x^c)` is the Reconstruction Error Ratio (RER)

### Interpretation

**RER < 1**: Sample reconstructs better from its own class autoencoder than any other class
- Indicates the sample fits well within its assigned class
- Low RER suggests the label is likely correct

**RER ≈ 1**: Sample reconstructs equally well from multiple class autoencoders
- The sample is on a decision boundary
- Inherently difficult or ambiguous

**RER > 1**: Sample reconstructs better from a different class autoencoder
- Strong signal of potential mislabeling
- Or indicates the sample is an outlier within its class

### Dataset-Level Metrics

**Chi (χ)**: Mean RER across all samples
- Measures overall dataset difficulty
- Lower chi = easier dataset (clear class separation)
- Higher chi = harder dataset (overlapping classes)

**Eta (η)**: Estimated noise rate
- Fraction of samples with RER > 1
- Approximates label noise in the dataset

### Why This Works

The key insight: if class-conditional data distributions differ meaningfully, a generative model (autoencoder) trained on one class will fail to reconstruct samples from other classes. The ratio of errors provides a normalized, comparable difficulty metric.

## Self-Supervised Learning with RER

### The Challenge

Traditional RER analysis requires labeled data to train K separate autoencoders. But what if we don't have labels?

### The Solution: Prototype-Guided Assignment

Instead of using labels, use reconstruction errors themselves to assign samples to classes:

1. **Initialize** with a few prototypes per class (can be as few as 1)
2. **Forward pass** through all K autoencoders for each sample
3. **Assign** sample to the class with minimum reconstruction error
4. **Train** each autoencoder only on samples assigned to it
5. **Refine** assignments iteratively as autoencoders improve

### Why Prototypes?

Prototypes serve as anchors:
- Prevent autoencoders from collapsing to identical models
- Provide initial class structure in the feature space
- Enable semi-supervised learning with very few labeled examples

Without prototypes, the system degenerates: all autoencoders converge to the same model that reconstructs all data equally well.

### Prototype Selection Strategies

**Random**: Select N random samples per class
- Fast and simple
- Works well with good feature representations

**Centroid**: Select samples closest to class centroids (requires initial labels)
- More stable initialization
- Better represents class structure

**First**: Select first N samples encountered per class
- Deterministic
- Good for reproducibility

**External**: Load prototypes from file or folder
- Fully deterministic and reproducible
- Supports variable prototype counts per class (e.g., 3 for class 0, 2 for class 1)
- Two formats supported:
  - `.pth` file: `{"images": Tensor[N, C, H, W], "class_ids": Tensor[N]}`
  - Folder: class subfolders containing prototype images (`0/`, `1/`, etc.)

### Anchoring Loss

To prevent autoencoder drift, add a small anchoring loss:

```
L_anchor = ||Δ^c(prototypes_c) - 0||²
```

This keeps each autoencoder specialized on its prototype samples, preventing homogenization across classes.

## Current Research Progress

### Experimental Setup

**Dataset**: MNIST digits (10 classes, 28x28 grayscale images)

**Feature Extraction**:
- OMNIGLOT-pretrained autoencoder embeddings (128-dim)
- Pre-computed and cached for fast training
- Alternative: DINOv2 features (384-dim) also show promise

**Model Architecture**:
- K=10 shallow autoencoders (one per digit)
- Encoder: [128 → 256 → 10] (bottleneck)
- Decoder: [10 → 256 → 128]
- Total: ~2M parameters across all autoencoders

**Training**:
- 100 epochs
- Batch size: 128
- Learning rate: 1e-3
- Prototype settings: 3 prototypes per class (centroid selection)
- RER threshold: 2.0
- Prototype anchoring weight: 0.1

### Results

**Validation Accuracy: 85%**

Without using any labels during training (only prototypes for initialization), the system achieves 85% classification accuracy by assigning samples based purely on reconstruction errors.

**Key Observations**:

1. **Few prototypes suffice**: Even 1-3 prototypes per class enable successful clustering
2. **Feature quality matters**: Pre-trained embeddings (MAE, DINO) work much better than raw pixels
3. **Convergence is fast**: Most assignments stabilize within 20-30 epochs
4. **Threshold tuning**: RER threshold of 2.0 provides good balance between precision and coverage

### Promising Directions

**Multi-view consistency**: Samples should reconstruct consistently from the same class autoencoder across different augmentations

**Active learning**: High-RER samples (near decision boundaries) are ideal candidates for human labeling

**Hierarchical clustering**: Use RER to build class hierarchies (which classes are most similar?)

**Transfer learning**: Train autoencoders on one dataset, apply to related domains

## Applications and Use Cases

### 1. Unsupervised Clustering

Cluster samples without any labels. Useful when:
- Labeling is prohibitively expensive
- Class structure is unknown
- Exploring new datasets

### 2. Semi-Supervised Learning

Bootstrap from a few labeled examples (prototypes) to label entire dataset. Particularly effective when:
- Budget allows labeling only 1-10 samples per class
- Domain expertise is scarce
- Rapid iteration is needed

### 3. Noisy Label Learning

Learn robust representations despite label noise. Applications:
- Web-scraped datasets with unreliable labels
- Crowdsourced annotations with errors
- Legacy datasets with known quality issues

### 4. Mislabel Detection

Identify potentially mislabeled samples for human review. Use when:
- Dataset quality is critical
- Re-labeling entire dataset is infeasible
- Prioritizing cleaning efforts

### 5. Dataset Difficulty Estimation

Characterize dataset difficulty before committing resources. Helps with:
- Deciding modeling complexity needed
- Estimating annotation budget
- Comparing dataset versions

### 6. Active Learning

Select most informative samples for labeling based on RER. Strategies:
- High RER samples (decision boundaries)
- Low confidence assignments
- High variance across classes

## Theoretical Considerations

### Assumptions

1. **Class-conditional distributions differ**: Classes must have distinct structure in feature space
2. **Features are informative**: Raw pixels rarely work; pre-trained features are essential
3. **Autoencoders have sufficient capacity**: Must be able to model within-class variation
4. **Initialization matters**: Prototypes must represent different classes

### Limitations

**Feature dependence**: Performance strongly depends on feature extractor quality. Poor features yield poor results.

**Class imbalance**: Highly imbalanced datasets can bias assignments toward majority classes. RER partially mitigates this but doesn't eliminate it.

**Overlapping classes**: When classes are nearly indistinguishable (e.g., fine-grained recognition), RER values approach 1 for all samples.

**Prototype quality**: Bad prototype selection can permanently bias the system. Centroid selection is more robust than random.

**Computational cost**: Training K autoencoders scales linearly with number of classes. For very large K (e.g., 1000+ classes), this becomes expensive.

### Comparison to Alternatives

**vs. K-means clustering**:
- RER provides sample-level difficulty scores, not just assignments
- Handles non-convex clusters better
- More interpretable (reconstruction error has semantic meaning)

**vs. Deep clustering (e.g., SCAN, SwAV)**:
- Simpler architecture (shallow autoencoders)
- Faster training
- More interpretable metrics
- Explicitly models class-conditional distributions

**vs. Semi-supervised methods (e.g., FixMatch)**:
- Doesn't require consistency regularization
- Works with any feature extractor
- Provides uncertainty estimates (RER)

## Future Work

### Methodological Extensions

1. **Dynamic prototype refinement**: Replace initial prototypes with class centroids as assignments stabilize

2. **Hierarchical RER**: Multi-level class structure (coarse to fine-grained)

3. **Multi-task RER**: Joint training of autoencoders with auxiliary tasks (rotation prediction, colorization)

4. **Adaptive thresholds**: Learn per-class RER thresholds instead of global constant

5. **Prototype-free initialization**: Explore clustering methods that don't require any labels

### Applications

1. **Document clustering**: Apply to text embeddings for topic modeling without labels

2. **Medical imaging**: Cluster pathology slides for rare disease discovery

3. **Anomaly detection**: High RER → potential anomaly or out-of-distribution sample

4. **Continual learning**: Use RER to detect distribution shift and new classes

5. **Dataset distillation**: Select most representative samples per class (low RER)

### Theoretical Analysis

1. **Convergence guarantees**: Under what conditions does assignment converge?

2. **Sample complexity**: How many prototypes are needed as a function of dataset size?

3. **Identifiability**: Can we recover true labels from RER alone?

4. **Connections to information theory**: Relate RER to mutual information between features and labels

## Related Work

**Reconstruction-based methods**:
- Autoencoders for anomaly detection (Sakurada & Yairi, 2014)
- One-class SVMs for outlier detection

**Deep clustering**:
- DeepCluster (Caron et al., 2018)
- SCAN (Van Gansbeke et al., 2020)
- SwAV (Caron et al., 2020)

**Semi-supervised learning**:
- Pseudo-labeling (Lee, 2013)
- FixMatch (Sohn et al., 2020)

**Dataset characterization**:
- Data Maps (Swayamdipta et al., 2020)
- Influence functions (Koh & Liang, 2017)

**RER foundations**:
- Marks et al. (2024): Original RER framework

## Conclusion

Reconstruction Error Ratios provide a principled, interpretable metric for dataset analysis that bridges supervised and unsupervised learning. By leveraging reconstruction errors from class-wise autoencoders, we can:

- Estimate dataset difficulty without extensive training
- Detect mislabels automatically
- Cluster and classify with minimal supervision
- Identify informative samples for labeling

Current results (85% accuracy on MNIST with 3 prototypes per class) demonstrate the viability of prototype-guided self-supervised learning. The framework is modular, interpretable, and applicable across domains.

The path forward involves scaling to larger datasets, exploring hierarchical structures, and developing theoretical understanding of convergence and sample complexity.

## References

Marks, M., et al. (2024). "Reconstruction Error Ratios for Dataset Analysis."

Caron, M., et al. (2018). "Deep Clustering for Unsupervised Learning of Visual Features."

Van Gansbeke, W., et al. (2020). "SCAN: Learning to Classify Images without Labels."

Swayamdipta, S., et al. (2020). "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics."
