# Oobleck 
It's a working title. Possible backcronyms:
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
- visqol and si-snr expert listeners for a subjective performance measure
- FAD?
- ...
- (waiting for the first models to be trained)

We will be experimenting with different loss functions 
- perceptual loss, etc
- place loss functions into [auraloss](https://github.com/csteinmetz1/auraloss) and import it

## Usage

Instantiation of an OOBLECK autoencoder corresponding to a given `.gin` file can be done following 

```python
import gin
import torch

from oobleck import AudioAutoEncoder

gin.parse_config_file("oobleck/configs/debug/base.gin")
model = AudioAutoEncoder()

inputs = {"input": torch.randn(1, 1, 2**16)}
loss, outputs = model.loss(inputs)

for k, v in outputs.items():
    print(f"{k}.shape = {v.shape}")

# >>> input.shape = torch.Size([1, 1, 65536])
# >>> latent.shape = torch.Size([1, 128, 256])
# >>> output.shape = torch.Size([1, 1, 65536])

print(loss)

# >>> tensor(0.8029, grad_fn=<MeanBackward0>)
```
