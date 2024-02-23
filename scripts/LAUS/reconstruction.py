import os
import  torch

def load_gen_model(model_dir=None, epoch=None, num_downs=8, norm_layer=nn.BatchNorm2d, device=None):
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





model_gen, _, _, _, _ = load_gen_model(model_dir=os.path.join(this_dir, "models", 'wang'),
                                         epoch=epoch,
                                         num_downs=5,
                                         device=device)

model_gen.eval()

## Code to read channel data (2, 800, 128)
##

channel_data = channel_data / channel_data.abs().max()
N, C, H, W = channel_data.size()
z = torch.randn(N, 1, H, W).to(device)
wang_phantom = model_gen(torch.cat((channel_data, z), dim=1))