import numpy as np
import torch 
import torch.nn as nn 
from pytorch_msssim import MS_SSIM

def calculate_smape_score(model: nn.Module, input_path: str, target_path: str) -> float:
    """
    Calcula el Symmetric Mean Absolute Percentage Error (sMAPE) y lo convierte a un score.
    
    Args:
        model: Modelo a evaluar
        input_path: Ruta al archivo de entrada
        target_path: Ruta al archivo objetivo
        
    Returns:
        float: Score basado en sMAPE, entre 0 y 1 donde 1 es mejor
    """
    sample = np.load(input_path)
    input_tensor = torch.tensor(sample, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    target_np = np.load(target_path)
    target_tensor = torch.from_numpy(target_np).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    
    # print("Target range:", target_tensor.min().item(), target_tensor.max().item())
    # print("Output range:", output.min().item(), output.max().item())
    
    # Calcular sMAPE (Symmetric Mean Absolute Percentage Error)
    numerator = torch.abs(target_tensor - output)
    denominator = (torch.abs(target_tensor) + torch.abs(output)) / 2
    smape = torch.mean(numerator / denominator) * 100

    # print("Raw sMAPE:", smape.item())
    
    # Convertir a score - el sMAPE estará entre 0 y 100
    score = 1 / (1 + smape/100)
    # print("Final score:", score.item())
    
    return score.item()


def calculate_ms_ssim(model: nn.Module, input_path: str, target_path: str) -> float:
    """
    Multi-Scale Structural Similarity Index
    """
    sample = np.load(input_path)
    input_tensor = torch.tensor(sample, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    #Agregamos la dimension latent space z
    # N, _, H, W = input_tensor.size()
    # z = torch.randn(N, 1, H, W).to(device)
    # z = torch.load('/content/latent_z.pth').to(device)
    # input_tensor = torch.cat([input_tensor, z], dim=1)

    with torch.no_grad():
        output = model(input_tensor)

    # Cargar el target y convertirlo a tensor
    target_np = np.load(target_path)
    target_tensor = torch.from_numpy(target_np).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    data_range = max(output.max(), target_tensor.max()) - min(output.min(), target_tensor.min())

    ms_ssim_loss = MS_SSIM(data_range=data_range, size_average=True, channel=1, win_size=7, K = (0.01, 0.03))
    # print('min target', target_tensor.min(), '\n')
    # print('max target', target_tensor.max(), '\n')
    # print('min output', output.min(), '\n')
    # print('max output', output.max(), '\n')

    ms_ssim_value = ms_ssim_loss(output, target_tensor)

    return ms_ssim_value.cpu().item()

def calculate_mse(model: nn.Module, input_path: str, target_path: str) -> float:
    sample = np.load(input_path)
    input_tensor = torch.tensor(sample, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    target_np = np.load(target_path)
    target_tensor = torch.from_numpy(target_np).unsqueeze(dim=0).unsqueeze(dim=0).to(device)

    mse_loss = nn.MSELoss()
    mse_value = mse_loss(output, target_tensor)

    return mse_value.cpu().item()

def calculate_gcnr(model, input_path: str) -> float:
    """
    Calcula el Generalized Contrast to Noise Ratio
    """
    try:
        # Cargar y procesar input
        sample = np.load(input_path)
        input_tensor = torch.tensor(sample, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # Obtener predicción
        with torch.no_grad():
            output = model(input_tensor)
            
        # Extraer ROIs
        inclusion = output[:, :, 128:228, 39:61].squeeze().cpu().numpy()
        back = output[:, :, 670:770, 94:116].squeeze().cpu().numpy()

        # Verificar que las ROIs no estén vacías
        if inclusion.size == 0 or back.size == 0:
            print("ROIs vacías detectadas")
            return 0.0

        # Calcular histogramas con bins fijos para consistencia
        bins = np.linspace(0, 1, 257)  # 256 bins entre 0 y 1
        f, _ = np.histogram(inclusion, bins=bins, density=True)
        g, _ = np.histogram(back, bins=bins, density=True)

        # Normalizar histogramas
        f = f / (f.sum() + 1e-10)
        g = g / (g.sum() + 1e-10)

        # Calcular gCNR
        gcnr_value = 1 - np.sum(np.minimum(f, g))

        # Verificar valor válido
        if not (0 <= gcnr_value <= 1):
            print(f"gCNR fuera de rango: {gcnr_value}")
            return 0.0

        return gcnr_value

    except Exception as e:
        print(f"Error en calculate_gcnr: {str(e)}")
        return 0.0
