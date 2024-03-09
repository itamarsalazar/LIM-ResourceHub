import torch
import os
from pathlib import Path
# from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import guided_diffusion_v3 as gd
from datetime import datetime
import torch.nn.functional as func
from model7 import UNETv13
# from model4 import UNETv10_5, UNETv10_5_2
import torch.nn as nn

from torchvision import transforms as T
import PIL
from PIL import Image

TRAIN_PATH = '/TESIS/DATOS_1/rf_train'
TRAIN_ENH_PATH= '/TESIS/DATOS_1/target_enh/'
TRAIN_ONEPW_PATH= '/TESIS/DATOS_TESIS2/onepw_train'

'''
DATASET
'''
#creating our own Dataset
#esta clase va a heredar de la clase Dataset de Pytorch
class ONEPW_Dataset(Dataset):
    def __init__(self, data, onepw_img=None, img_transforms=None, onepw_img_transforms=None):
        '''
        data - train data path
        enh_img - train enhanced images path
        '''
        self.train_data = data
        self.train_onepw_img = onepw_img
        self.img_transforms = img_transforms
        self.onepw_img_transforms = onepw_img_transforms

        self.images = sorted(os.listdir(self.train_data))
        self.onepw_images = sorted(os.listdir(self.train_onepw_img))
  
    #regresar la longitud de la lista, cuanto selementos hay en el dataset
    def __len__(self):
        if self.onepw_images is not None:
            assert len(self.images) == len(self.onepw_images), 'not the same number of images ans enh_images'
        return len(self.images)

    def __getitem__(self, idx):
        rf_image_name = os.path.join(self.train_data, self.images[idx])
        rf_image = np.load(rf_image_name)
        trans = T.ToTensor()

        if self.img_transforms is not None:
            rf_image = Image.fromarray(rf_image)
            rf_image = self.img_transforms(rf_image)
        else:
            #rf_image = trans(np.float32(rf_image))
            rf_image = trans(np.float32(rf_image)).unsqueeze(0)
            #rf_image = F.interpolate(rf_image, size=(64,64), mode='bilinear', align_corners=False)
            rf_image = rf_image.squeeze(0)
            #img = F.interpolate(img, size=(64,64), mode='bilinear', align_corners=False)
            #img = img.squeeze(0)
            #img = trans(img)


        if self.train_onepw_img is not None:
            onepw_image_name = os.path.join(self.train_onepw_img, self.onepw_images[idx])
            onepw_img = np.load(onepw_image_name)
            if self.onepw_img_transforms is not None:
                onepw_img = Image.fromarray(onepw_img)
                onepw_img = self.onepw_img_transforms(onepw_img)
            else:
                #enh_img = trans(np.float32(enh_img))
                onepw_img = trans(np.float32(onepw_img)).unsqueeze(0)
                #enh_img = F.interpolate(enh_img, size=(64,64), mode='bilinear', align_corners=False)
                onepw_img = onepw_img.squeeze(0)
                # new_min = -1
                # new_max = 1
                # enh_img = enh_img * (new_max - new_min) + new_min
                #enh_img = F.interpolate(enh_img, size=(64,64), mode='bilinear', align_corners=False)
                #enh_img = enh_img.squeeze(0)
                #enh_img = trans(enh_img)
        else:
            return rf_image

        return rf_image, onepw_img

def main():
    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)
    # save_dir = Path(os.getcwd())/'weights'/'v13'
    save_dir = '/CODIGOS_TESIS/TESIS2/Unet_models/v15'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # training hyperparameters
    batch_size = 4  # 4 for testing, 16 for training
    n_epoch = 10
    l_rate = 1e-5  # changing from 1e-5 to 1e-6, new lr 1e-7

    # Loading Data
    # input_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\input_overfit'
    # output_folder = r'C:\Users\sebas\Documents\Data\DiffusionBeamformer\target_overfit'
    # input_folder = r'C:\Users\u_imagenes\Documents\smerino\training\input'
    # output_folder = r'C:\Users\u_imagenes\Documents\smerino\training\target_enh'


    train_dataset = ONEPW_Dataset(TRAIN_PATH, TRAIN_ONEPW_PATH)

    BATCH_SIZE = 4

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)

    #######################
    # dataset = gd.CustomDataset(input_folder, output_folder, transform=True)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Dataloader length: {len(train_loader)}')
    #######################

    for i, (x, y) in enumerate(train_loader):
        print(i, x.shape,y.shape)
        if i==9: break

    # DDPM noise schedule
    time_steps = 1000
    betas = gd.get_named_beta_schedule('linear', time_steps)
    diffusion = gd.SpacedDiffusion(
        use_timesteps = gd.space_timesteps(time_steps, section_counts=[time_steps]),
        betas = betas,
        model_mean_type = gd.ModelMeanType.EPSILON,
        model_var_type= gd.ModelVarType.FIXED_LARGE,
        loss_type = gd.LossType.MSE,
        rescale_timesteps = True,
    )
    
    # Model and optimizer
    nn_model = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
    # nn_model = UNETv10_5_2(emb_dim=64*4).to(device)
    print("Num params: ", sum(p.numel() for p in nn_model.parameters() if p.requires_grad))

    optim = torch.optim.Adam(nn_model.parameters(), lr=l_rate)

    trained_epochs = 0
    if trained_epochs > 0:
        nn_model.load_state_dict(torch.load(save_dir+f"/model_{trained_epochs}.pth", map_location=device))  # From last model
        loss_arr = np.load(save_dir+f"/loss_{trained_epochs}.npy").tolist()  # From last model
    else:
        loss_arr = []

    # Training
    nn_model.train()
    # pbar = tqdm(range(trained_epochs+1,n_epoch+1), mininterval=2)
    print(f' Epoch {trained_epochs}/{n_epoch}, {datetime.now()}')
    for ep in range(trained_epochs+1, n_epoch+1):
        # pbar = tqdm(train_loader, mininterval=2)
        for x, y in train_loader:  # x: images
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # perturb data
            noise = torch.randn_like(y)
            t = torch.randint(0, time_steps, (x.shape[0],)).to(device)
            y_pert = diffusion.q_sample(y, t, noise)
            
            # use network to recover noise
            predicted_noise = nn_model(x, y_pert, t)

            # loss is mean squared error between the predicted and true noise
            loss = func.mse_loss(predicted_noise, noise)
            loss.backward()

            # nn.utils.clip_grad_norm_(nn_model.parameters(),0.5)
            loss_arr.append(loss.item())
            optim.step()
            torch.save(nn_model.state_dict(), save_dir+f"/model_{ep}.pth")

        print(f' Epoch {ep:03}/{n_epoch}, loss: {loss_arr[-1]:.2f}, {datetime.now()}')
        # save model every x epochs
        if ep % 5 == 0 or ep == int(n_epoch) or ep == 0:
            torch.save(nn_model.state_dict(), save_dir+f"/model_{ep}.pth")
            np.save(save_dir+f"/loss_{ep}.npy", np.array(loss_arr))

if __name__ == '__main__':
    main()
