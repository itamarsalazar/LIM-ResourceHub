# Compute contrast ratio
def contrast(img1, img2):
    return 20 * np.log10(img1.mean() / img2.mean())
    # return 20 * np.log10(img1.median() / img2.median())


# Compute contrast-to-noise ratio
def cnr(img1, img2):
    return (img1.mean() - img2.mean()) / np.sqrt(img1.var() + img2.var())


# Compute the generalized contrast-to-noise ratio
def gcnr(img1, img2):
    _, bins = np.histogram(np.concatenate((img1, img2)), bins=256)
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))


def snr(img):
    return img.mean() / img.std()


def psnr(img1, img2):
    dynamic_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    # dynamic_range = 60
    return 20 * np.log10(dynamic_range / l2loss(img1, img2))


if __name__ == '__main__':
    distances = np.sqrt((grid_full[:, :, 0] - xctr) ** 2 + (grid_full[:, :, 2] - zctr) ** 2)
    r1 = r - 1e-3
    r2 = r + 1.5e-3
    r3 = np.sqrt(r1 ** 2 + r2 ** 2)
    roi_in = distances < r1
    roi_out = distances > r2 & distances<r3 # verificar
    roi_cyst = distances <= r  # auxiliar variable

    bmode = model(channel_data)
    # bmode ->  numpy array (800x128)
    env = 10**(bmode/20)
    env_in = env[roi_in]
    env_out = env[roi_out]
    dct['contrast'] = contrast(env_in, env_out)
    dct['cnr'] = cnr(env_in, env_out)
    dct['gcnr'] = gcnr(env_in, env_out)
    dct['snr'] = snr(env_out)
    # dct['psnr'] = psnr(bmode, bmode_ref)


