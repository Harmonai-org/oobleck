import torchaudio


def encoder_out_size(latent_size: int, mode: str):
    if mode == "vae": latent_size = latent_size * 2
    return latent_size


def get_spectrogram(n_fft: int):
    return torchaudio.transforms.Spectrogram(
        n_fft,
        hop_length=n_fft // 4,
        power=None,
        normalized=True,
        center=False,
        pad_mode=None,
    )
