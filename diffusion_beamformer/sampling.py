import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from plotting_utils import *
import torch
import numpy as np

import guided_diffusion_v3 as gd
from torchvision import transforms


def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

############

diffusion = create_gaussian_diffusion()
input_folder = '/TESIS/DATOS_1/rf_train'
output_folder = '/TESIS/DATOS_TESIS2/onepw_train'

data = gd.CustomDataset(input_folder, output_folder, transform=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

BATCH_SIZE = 1
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
x_sample, y_sample = next(iter(dataloader))
x_sample = x_sample.to(device)
y_sample = y_sample.to(device)

##################

from model7 import UNETv13

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
save_dir = '/CODIGOS_TESIS/TESIS2/Unet_models/ddpm_tmodels/v15'
training_epochs = 10
model13A = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
model13A.load_state_dict(torch.load(f"{save_dir}/model_{training_epochs}.pth", map_location=device))
print("Num params: ", sum(p.numel() for p in model13A.parameters()))

plt.figure(figsize=(9, 3))
loss_npy = np.load(f"{save_dir}/loss_{training_epochs}.npy")
plt.scatter([x for x in range(len(loss_npy))], loss_npy, s=0.2)
plt.xlabel('# Iteration')
plt.ylabel('Loss')
#plt.xlim(000,35000)
plt.ylim(0, 0.01)
plt.show()

##################

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: (t * 60) - 60.),
        transforms.Lambda(lambda t: t.numpy())
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image), cmap='gray', extent=[-20,20,50,0])
    plt.clim(-60,0)

def show_reverse_process(intermediate):
    """ Shows a list of tensors from the sampling process """
    num_intermediate = len(intermediate)
    plt.figure(figsize=(15,2))
    plt.axis('off')
    for id, y_gen in enumerate(intermediate):
        plt.subplot(1, num_intermediate, id+1)
        show_tensor_image(y_gen)
    plt.show()

intermediate = []
for step in diffusion.p_sample_loop_progressive(model13A, y_sample.shape, x_sample, progress=True, clip_denoised=True):
    intermediate.append(step['sample'].cpu().detach())
show_reverse_process(intermediate[::100])

plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
show_tensor_image(intermediate[-1])
plt.colorbar()
plt.title('Diffusion')

plt.subplot(1, 2, 2)
show_tensor_image(y_sample.cpu())
plt.colorbar()
plt.title('Objective')
plt.show()
