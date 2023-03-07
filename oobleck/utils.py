def encoder_out_size(latent_size: int, mode: str):
    if mode == "vae": latent_size = latent_size * 2
    return latent_size
