# Oobleck 
"The more you compress it, the harder it gets"

Possible backronyms:
- "Out-of-the-box Latent Encoder Construction Kit"
- "Over-optimized Latent Encoder Construction Kit"

*(Open to other names, we can vote at the end.)*

## What is it?
MIT-licensed soundstream-ish VAE audio codecs for downstream neural audio synthesis.

We will be creating at least 
-  a continuous VAE (for downstream audio diffusion, etc)
-  a vector quantized VAE (for downstream MusicLM, etc)
-  a spherical VAE (for quantum circuits, etc)

We will validate them using
- MUSHRA testing protocol with expert listeners for a subjective performance measure
- visqol and si-snr expert listeners for an objective performance measure
- FAD?
- ...
- (waiting for the first models to be trained)

We will be experimenting with different loss functions 
- perceptual loss, etc
- place loss functions into [auraloss](https://github.com/csteinmetz1/auraloss) and import it

## Installation

```bash
OOBLECK_VERSION=develop pip install git+https://github.com/Harmonai-org/oobleck.git
```

## Usage

Instantiation of an OOBLECK autoencoder corresponding to a given `.gin` file can be done following 

```python
import gin
import torch

from oobleck import AudioAutoEncoder

gin.parse_config_file("base/base.gin")
model = AudioAutoEncoder()

inputs = {"waveform": torch.randn(1, 1, 2**16)}
outputs = model.loss(inputs)

print(outputs.keys())

# >>> dict_keys(['waveform', 'latent', 'reconstruction', \
# >>> 'score_waveform', 'score_reconstruction', 'features_reconstruction', \
# >>> 'features_waveform', 'generator_loss', 'MultiResolutionSTFTLoss', \
# >>> 'discriminator_loss'])
```
