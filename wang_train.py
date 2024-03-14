import os
import sys

import torch

sys.path.insert(0, '/nfs/privileged/isalazar/projects/ultrasound-image-formation/')
import pickle
import numpy as np

from pytorch_msssim import MS_SSIM

from train_utils import splitDataloader, save_model, save_history, create_dir

from src.models.unet.unet_model import *
from src.models.dataloaders.cystDataset import CystDatasetTUFFC_Wang

from torchvision.transforms import Resize

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def load_wang_model(model_dir=None, epoch=None, num_downs=8, norm_layer=nn.BatchNorm2d, device=None):
    # Instantiate the model architecture
    generator = Wang2020UnetGenerator(input_nc=3, #channel data (2) + latent space z (1)
                                      output_nc=1,
                                      num_downs=num_downs,
                                      ngf=64,
                                      norm_layer=norm_layer,
                                      use_dropout=False).to(device)

    discriminator = Wang2020UnetDiscriminator(input_nc=3, # channel data(2) + real or fake bmode (1)
                                              ndf=64,
                                              n_layers=3,
                                              norm_layer=norm_layer).to(device)

    if isinstance(epoch, int) and epoch == -1:
        gen_history = {"train_loss": [], "val_loss": []}
        genL1_history = {"train_loss": [], "val_loss": []}
        disc_history = {"train_loss": [], "val_loss": []}
        print("Model initialized with random weights")
    else:
        if isinstance(epoch, int):
            generator_filename = os.path.join(model_dir, 'model_gen_%.4d.t7' % epoch)
            discriminator_filename = os.path.join(model_dir, 'model_disc_%.4d.t7' % epoch)
            gen_history_filename = os.path.join(model_dir, 'history_gen_%.4d.pkl' % epoch)
            genL1_history_filename = os.path.join(model_dir, 'history_genL1_%.4d.pkl' % epoch)
            disc_history_filename = os.path.join(model_dir, 'history_disc_%.4d.pkl' % epoch)
            print(f"Loading models {epoch}...")
        elif isinstance(epoch, str):
            generator_filename = os.path.join(model_dir, 'model_gen_%s.t7' % epoch)
            discriminator_filename = os.path.join(model_dir, 'model_disc_%s.t7' % epoch)
            gen_history_filename = os.path.join(model_dir, 'history_gen_%s.pkl' % epoch)
            genL1_history_filename = os.path.join(model_dir, 'history_genL1_%s.pkl' % epoch)
            disc_history_filename = os.path.join(model_dir, 'history_disc_%s.pkl' % epoch)

            print(f"Loading models {epoch}...")
        else:
            raise ValueError("Invalid epoch type. Should be either int or str.")

        # If weights path is provided, load the weights from the saved model
        if (os.path.isfile(generator_filename) and os.path.isfile(gen_history_filename)
                and os.path.isfile(genL1_history_filename)):
            checkpoint = torch.load(generator_filename, map_location=device)
            generator.load_state_dict(checkpoint)
            with open(gen_history_filename, 'rb') as f:
                gen_history = pickle.load(f)
            with open(genL1_history_filename, 'rb') as f:
                genL1_history = pickle.load(f)
            print(f"Generator {epoch} loaded.")
        else:
            raise ValueError(f" (Generator) No weights or history found at {model_dir}.")

        # If weights path is provided, load the weights from the saved model
        if os.path.isfile(discriminator_filename) and os.path.isfile(disc_history_filename):
            checkpoint = torch.load(discriminator_filename, map_location=device)
            discriminator.load_state_dict(checkpoint)
            with open(disc_history_filename, 'rb') as f:
                disc_history = pickle.load(f)
            print(f"Discriminator {epoch} loaded.")
        else:
            raise ValueError(f" (Discriminator) No weights or history found at {model_dir}.")

    return generator, discriminator, gen_history, genL1_history, disc_history


def load_wang_history(model_dir=None, epoch=None):

    if isinstance(epoch, int) and epoch == -1:
        gen_history = {"train_loss": [], "val_loss": []}
        genL1_history = {"train_loss": [], "val_loss": []}
        disc_history = {"train_loss": [], "val_loss": []}
        print("History initialized as empty dictionary")
    else:
        if isinstance(epoch, int):
            gen_history_filename = os.path.join(model_dir, 'history_gen_%.4d.pkl' % epoch)
            genL1_history_filename = os.path.join(model_dir, 'history_genL1_%.4d.pkl' % epoch)
            disc_history_filename = os.path.join(model_dir, 'history_disc_%.4d.pkl' % epoch)
            print(f"Loading histories {epoch}...")
        elif isinstance(epoch, str):
            gen_history_filename = os.path.join(model_dir, 'history_gen_%s.pkl' % epoch)
            genL1_history_filename = os.path.join(model_dir, 'history_genL1_%s.pkl' % epoch)
            disc_history_filename = os.path.join(model_dir, 'history_disc_%s.pkl' % epoch)

            print(f"Loading histories {epoch}...")
        else:
            raise ValueError("Invalid epoch type. Should be either int or str.")

        # If weights path is provided, load the weights from the saved model
        if (os.path.isfile(gen_history_filename)
                and os.path.isfile(genL1_history_filename)):
            with open(gen_history_filename, 'rb') as f:
                gen_history = pickle.load(f)
            with open(genL1_history_filename, 'rb') as f:
                genL1_history = pickle.load(f)
            print(f"Generator {epoch} loaded.")
        else:
            raise ValueError(f" (Generator) No history found at {model_dir}")

        # If weights path is provided, load the weights from the saved model
        if os.path.isfile(disc_history_filename):
            with open(disc_history_filename, 'rb') as f:
                disc_history = pickle.load(f)
            print(f"Discriminator {epoch} loaded.")
        else:
            raise ValueError(f" (Discriminator) No history found at {model_dir}.")

    return gen_history, genL1_history, disc_history


def train(models, prev_hists, train_loader, val_loader, init_epoch, last_epoch, model_dir, loss_fn, device):
    gen, disc = models
    gen_prev_hist, genL1_prev_hist, disc_prev_hist = prev_hists

    lr = 0.0002
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    criterionGAN, criterionL1, lambda_L1 = loss_fn


    gen_history = {"train_loss": [*gen_prev_hist['train_loss']],
                   "val_loss": [*gen_prev_hist['val_loss']]}
    genL1_history = {"train_loss": [*genL1_prev_hist['train_loss']],
                     "val_loss": [*genL1_prev_hist['val_loss']]}
    disc_history = {"train_loss": [*disc_prev_hist['train_loss']],
                    "val_loss": [*disc_prev_hist['val_loss']]}

    for epoch in range(init_epoch, last_epoch):
        gen_train_loss_sum = 0
        genL1_train_loss_sum = 0
        disc_train_loss_sum = 0
        train_n = 0

        gen.train()
        disc.train()
        for batch_idx, samples in enumerate(train_loader):
            channel_data, real_bmode = samples['input'], samples['target']
            channel_data, real_bmode = channel_data.to(device), real_bmode.to(device)
            # print(f"(Sizes) channel_data: {channel_data.size()}, real_bmode: {real_bmode.size()}")
            N, C, H, W = channel_data.size()
            z = torch.randn(N, 1, H, W).to(device)
            print(f"(Size) channel_data: {channel_data.size()}")
            print(f"(Size) z: {z.size()}")
            print(f"(Size) cat: {torch.cat((channel_data, z), dim=1).size()}")

            gen.train()
            disc.train()

            fake_bmode = gen(torch.cat((channel_data, z), dim=1))

            # Update D
            set_requires_grad(disc, True)           # enable backprop for D
            optimizer_D.zero_grad()                             # set D's gradients to zero
            fake_pair = torch.cat((channel_data, fake_bmode), 1)   # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = disc(fake_pair.detach())                # Fake; stop backprop to the generator by detaching fake_B
            aux = torch.tensor(0.0).expand_as(pred_fake).to(device)
            loss_D_fake = criterionGAN(pred_fake, aux)
            real_pair = torch.cat((channel_data, real_bmode), 1)
            pred_real = disc(real_pair)
            aux = torch.tensor(1.0).expand_as(pred_real).to(device)
            loss_D_real = criterionGAN(pred_real, aux)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()                                  # update D's weights

            # update G
            set_requires_grad(disc, False)          # D requires no gradients when optimizing G
            optimizer_G.zero_grad()                             # set G's gradients to zero
            fake_pair = torch.cat((channel_data, fake_bmode), 1) # First, G(A) should fake the discriminator
            pred_fake = disc(fake_pair)
            aux = torch.tensor(1.0).expand_as(pred_fake).to(device)
            loss_G = criterionGAN(pred_fake, aux)
            loss_G_L1 = criterionL1(fake_bmode, real_bmode) * lambda_L1     # Second, G(A) = B
            loss_G_compound = loss_G + loss_G_L1     # combine loss and calculate gradients
            loss_G_compound.backward()
            optimizer_G.step()                      # update G's weights

            # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # opt.step()

            disc_train_loss_sum += loss_D.item() * channel_data.size(0)
            gen_train_loss_sum += loss_G.item() * channel_data.size(0)
            genL1_train_loss_sum += loss_G_L1.item() * channel_data.size(0)

            train_n += channel_data.size(0)

            disc_history["train_loss"].append(loss_D.item())
            gen_history["train_loss"].append(loss_G.item())
            genL1_history["train_loss"].append(loss_G_L1.item())

            if batch_idx % 50 == 0:
                batch_str = f'\nEpoch {epoch}: batch {batch_idx} of {len(train_loader) - 1} '\
                            f'| disc loss: {disc_train_loss_sum / train_n :.6f}, '\
                            f'gen loss: {gen_train_loss_sum / train_n :.6f}, '\
                            f'genL1 loss: {genL1_train_loss_sum / train_n :.6f} | '\
                            f'lr: {lr} '
                print(batch_str, end="")
                with open(os.path.join(model_dir, 'batch.txt'), 'w') as f:
                    # Redirect stdout to the file
                    sys.stdout = f
                    # Print some messages
                    print(batch_str, end="")
                    # Reset stdout back to the console
                    sys.stdout = sys.__stdout__
            print('.', end="")

        disc_train_loss = disc_train_loss_sum / train_n
        gen_train_loss = gen_train_loss_sum / train_n
        genL1_train_loss = genL1_train_loss_sum / train_n

        disc_val_loss, gen_val_loss, genL1_val_loss = test(gen, disc, val_loader, loss_fn)

        disc_history["val_loss"].append(disc_val_loss)
        gen_history["val_loss"].append(gen_val_loss)
        genL1_history["val_loss"].append(genL1_val_loss)

        epoch_str = f"End of Epoch: {epoch} | "\
                    f"Disc Loss: {disc_train_loss:.6f}, "\
                    f"Gen Loss: {gen_train_loss:.6f}, "\
                    f"GenL1 Loss: {genL1_train_loss:.6f} | "\
                    f"| Val:  "\
                    f"Disc {disc_val_loss:.6f}, "\
                    f"Gen {gen_val_loss:.6f}, "\
                    f"GenL1 {genL1_val_loss:.6f}"
        # Open a file for writing
        with open(os.path.join(model_dir, 'epoch.txt'), 'w') as f:
            # Redirect stdout to the file
            sys.stdout = f
            # Print some messages
            print(epoch_str)
            # Reset stdout back to the console
            sys.stdout = sys.__stdout__
        print(epoch_str)
        try:
            disc_state_dict = disc.module.state_dict()
            gen_state_dict = gen.module.state_dict()
        except AttributeError:
            disc_state_dict = disc.state_dict()
            gen_state_dict = gen.state_dict()

        save_model("disc_intermediate", disc_state_dict, model_dir)
        save_history("disc_intermediate", disc_history, model_dir)

        save_model("gen_intermediate", gen_state_dict, model_dir)
        save_history("gen_intermediate", gen_history, model_dir)
        save_history("genL1_intermediate", genL1_history, model_dir)

        if (epoch+1)%50 == 0:
            save_model('disc_%.4d' % int(epoch), disc_state_dict, model_dir)
            save_history('disc_%.4d' % int(epoch), disc_history, model_dir)

            save_model('gen_%.4d' % int(epoch), gen_state_dict, model_dir)
            save_history('gen_%.4d' % int(epoch), gen_history, model_dir)
            save_history('genL1_%.4d' % int(epoch), genL1_history, model_dir)

    save_model("disc_last", disc_state_dict, model_dir)
    save_history("disc_last", disc_history, model_dir)

    save_model("gen_last", gen_state_dict, model_dir)
    save_history("gen_last", gen_history, model_dir)
    save_history("genL1_last", genL1_history, model_dir)

    return gen, disc, gen_history, genL1_history, disc_history


def test(gen, disc, val_loader, loss_fn):

    device = next(gen.parameters()).device.type

    criterionGAN, criterionL1, lambda_L1 = loss_fn

    disc_test_loss_sum = 0
    gen_test_loss_sum = 0
    genL1_test_loss_sum = 0
    test_n = 0
    gen.eval()
    disc.eval()
    with torch.no_grad():
        for batch_idx, samples in enumerate(val_loader):
            channel_data, real_bmode = samples['input'], samples['target']
            channel_data, real_bmode = channel_data.to(device), real_bmode.to(device)
            N, C, H, W = channel_data.size()
            z = torch.randn(N, 1, H, W).to(device)

            fake_bmode = gen(torch.cat((channel_data, z), dim=1))
            fake_pair = torch.cat((channel_data, fake_bmode), 1)
            pred_fake = disc(fake_pair.detach())
            aux = torch.tensor(0.0).expand_as(pred_fake).to(device)
            loss_D_fake = criterionGAN(pred_fake, aux)
            real_pair = torch.cat((channel_data, real_bmode), 1)
            pred_real = disc(real_pair)
            aux = torch.tensor(1.0).expand_as(pred_real).to(device)
            loss_D_real = criterionGAN(pred_real, aux)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            aux = torch.tensor(1.0).expand_as(pred_fake).to(device)
            loss_G = criterionGAN(pred_fake, aux)
            loss_G_L1 = criterionL1(fake_bmode, real_bmode) * lambda_L1     # Second, G(A) = B

            disc_test_loss_sum += loss_D.item() * channel_data.size(0)
            gen_test_loss_sum += loss_G.item() * channel_data.size(0)
            genL1_test_loss_sum += loss_G_L1.item() * channel_data.size(0)

            test_n += channel_data.size(0)
        disc_loss = (disc_test_loss_sum / test_n)
        gen_loss = (gen_test_loss_sum / test_n)
        genL1_loss = (genL1_test_loss_sum / test_n)

    return disc_loss, gen_loss, genL1_loss


if __name__ == '__main__':
    this_dir = '/nfs/privileged/isalazar/projects/ultrasound-image-formation/exploration/Journal2023/'
    data_dir = '/mnt/workerXspace/isalazar/datasets/simulatedCystDataset/TUFFC'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ########################################################
    load_epoch = 49     # set to -1 to train from scratch
    last_epoch = 100
    BATCH_SIZE = 32
    ########################################################
    print('Device is: %s' % device)
    save_model_dir = os.path.join(this_dir, "models", 'wang')

    create_dir(save_model_dir)

    full_dataset = CystDatasetTUFFC_Wang(data_dir=data_dir,
                                         input_subdir='input_id',
                                         target_subdir='target_from_raw',
                                         transform=None)

    train_loader, val_loader = splitDataloader(full_dataset,
                                               batch_size=(BATCH_SIZE, BATCH_SIZE),
                                               train_val_split=None,
                                               indices_dir=os.path.join(this_dir, "models"),
                                               previous_indices=True)

    gen, disc, gen_history, genL1_history, disc_history = load_wang_model(model_dir=save_model_dir,
                                                                          epoch=load_epoch,
                                                                          num_downs=5,
                                                                          device=device)

    gen, disc, gen_history, genL1_history, disc_history = train(models=(gen, disc),
                                                                prev_hists=(gen_history, genL1_history, disc_history),
                                                                train_loader=train_loader,
                                                                val_loader=val_loader,
                                                                init_epoch=load_epoch+1,
                                                                last_epoch=last_epoch,
                                                                model_dir=save_model_dir,
                                                                loss_fn= (nn.BCEWithLogitsLoss(), torch.nn.L1Loss(), 100.0),
                                                                device=device)


