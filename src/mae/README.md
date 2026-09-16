# Multi-View Masked Autoencoder (MAE)

A modular, minimal MAE implementation for learning embeddings from handwritten character images.

## Architecture Overview

### Encoder (`encoder.py`)
- ViT-style patch embedding with configurable patch size
- Transformer blocks with pre-norm architecture
- Supports CLS token for global feature extraction
- Sin-cos positional embeddings (fixed or learnable)

### Decoder (`decoder.py`)
- Lightweight transformer decoder (1-2 layers)
- Learnable mask tokens
- Separate positional embeddings
- Linear projection to pixel values

### Masking (`masking.py`)
- Random patch masking with configurable ratio
- Clean interface: `(x) -> (x_masked, mask, ids_restore)`
- Easily extensible for block/grid masking

### Main Model (`mae_model.py`)
- Combines encoder, decoder, masking, and loss
- `forward()`: Training with masking and reconstruction
- `encode()`: Feature extraction without masking
- `visualize()`: Generate reconstruction visualizations

### Multi-View (`multiview_mae.py`)
- Processes multiple augmented views
- Independent masking per view
- Exposes latents for cross-view losses
- Supports shared masking option

## Quick Start

### Training Single-View MAE

```bash
conda activate mlresearch
PYTHONPATH=src python src/cli.py fit --config conf/experiment/mae_mnist_simple.yaml
```

### Training Multi-View MAE

```bash
PYTHONPATH=src python src/cli.py fit --config conf/experiment/mae_mnist_multiview.yaml
```

### Feature Extraction

```python
from feature_extractors import MAEFeatureExtractor

# Load from checkpoint
extractor = MAEFeatureExtractor(checkpoint_path="path/to/checkpoint.ckpt")

# Extract features
features = extractor.extract(images)  # [batch, 128]
```

## Configuration

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | 64 | Input image size |
| `patch_size` | 8 | Patch size (8x8 = 64 patches) |
| `embed_dim` | 128 | Encoder embedding dimension |
| `encoder_depth` | 2 | Number of encoder layers |
| `decoder_depth` | 1 | Number of decoder layers |
| `mask_ratio` | 0.75 | Fraction of patches to mask |
| `num_views` | 2 | Views per sample (multi-view) |

## Model Size

Default configuration (~500K parameters):
- Patches: 64 (8x8 grid)
- Encoder: ~400K params (128-dim, 2 layers, 4 heads)
- Decoder: ~100K params (64-dim, 1 layer)

## Extension Points

### Adding New Losses

Create a new loss in `losses/`:

```python
from mae.losses.reconstruction import MAELoss

class ContrastiveLoss(MAELoss):
    def forward(self, pred, target, mask):
        # Your contrastive loss implementation
        pass
```

### Adding Cross-View Loss

In `multiview_mae.py`, modify `forward()` to add:

```python
# After processing all views
latent_similarity = F.cosine_similarity(latents[0], latents[1])
cross_view_loss = (1 - latent_similarity).mean()
total_loss = reconstruction_loss + 0.1 * cross_view_loss
```

### Adding Teacher-Student (DINO-style)

Add momentum encoder to `MultiViewMAE`:

```python
self.teacher = copy.deepcopy(self.mae.encoder)
# In forward: update teacher with EMA
```

## Integration with RERC

The MAE can be used as a feature extractor for class-wise autoencoders:

```python
from feature_extractors import MAEFeatureExtractor
from task import JITClasswiseAETask

# Use MAE for just-in-time feature extraction
extractor = MAEFeatureExtractor(checkpoint_path="mae_checkpoint.ckpt")
task = JITClasswiseAETask(
    feature_extractor=extractor,
    num_classes=10,
    feature_dim=extractor.feature_dim,
)
```

## File Structure

```
mae/
├── __init__.py           # Package exports
├── encoder.py            # ViT encoder
├── decoder.py            # Lightweight decoder
├── masking.py            # Masking strategies
├── mae_model.py          # Main MAE model
├── multiview_mae.py      # Multi-view wrapper
├── losses/
│   ├── __init__.py
│   └── reconstruction.py # MSE loss
└── README.md             # This file
```
