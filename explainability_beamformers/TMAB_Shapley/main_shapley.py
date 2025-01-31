import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from typing import List, Tuple, Dict
from pytorch_msssim import MS_SSIM
import math
import time
import h5py
import json
import os
import copy
import random
from .models import *
from .utils import *

# Configuración de dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIR = '/home/mstevend/ACSI/GRUPO_2/dpb4_sergio/data'
MODEL_PATH = '/home/mstevend/ACSI/GRUPO_2/dpb4_sergio/simplebeamformerV2/mssim_simplebeamformerV2'
CHECKPOINT_BASE = 'shapley_simplev2_mssim_checkpoint'

def tmab_shapley(model: nn.Module,
                 num_filters: int,
                 num_samples: int,
                 truncation: float,
                 delta: float,
                 epsilon: float,
                 img_dir: str,
                 checkpoint_interval: int = 100,
                 resume_from_checkpoint: bool = False,
                 resume_sample: int = None) -> Dict[Tuple[str, int], float]:
    """
    Implementación del algoritmo TMAB-Shapley usando la métrica MS-SSIM
    
    Args:
        model: Modelo a evaluar
        num_filters: Número de filtros importantes a identificar
        num_samples: Número máximo de iteraciones
        truncation: Umbral de truncamiento para performance
        delta: Parámetro delta para límites de confianza
        epsilon: Tolerancia para convergencia
        img_dir: Directorio con imágenes de entrada/target
        base_dir: Directorio para guardar resultados
        checkpoint_interval: Cada cuántas iteraciones guardar checkpoint
        resume_from_checkpoint: Si continuar desde último checkpoint
    
    Returns:
        Dict con valores Shapley para cada filtro
    """
                   
    if resume_from_checkpoint:
        state = load_checkpoint(resume_sample)
        filters = state['filters']
        values = {tuple(eval(k)): v for k, v in state['values'].items()}
        counts = {tuple(eval(k)): v for k, v in state['counts'].items()}
        variances = {tuple(eval(k)): v for k, v in state['variances'].items()}
        start_sample = state['current_sample']
        original_mssim = state['original_mssim'] # Cambiar por la métrica que se esté usando
    else:
        filters = get_filters(model)
        values = {tuple(filter): 0.0 for filter in filters}
        counts = {tuple(filter): 0 for filter in filters}
        variances = {tuple(filter): 0.0 for filter in filters}
        start_sample = 0
        original_mssim = np.mean([calculate_ms_ssim(model, f'{img_dir}/input_id/simu{i:05d}.npy',
                                                f'{img_dir}/target_from_raw/simu{i:05d}.npy')
                                 for i in range(2, 6)]) # Cambiar por la métrica que se esté usando

    print(f"Total number of filters: {len(filters)}")
    print("First 5 filters:", filters[:5])
    print("Last 5 filters:", filters[-5:])

    for sample in range(start_sample, num_samples):
        print(f"Sample {sample + 1}/{num_samples}")
        permutation = np.random.permutation(filters).tolist()
        model_copy = SimpleBeamformer().to(device) # Cambiar al modelo que se esté usando
        model_copy.load_state_dict(model.state_dict())

        mssim_score = original_mssim
        for i, filter in enumerate(permutation):
            filter_tuple = filter
            filter_tuple[1] = int(filter_tuple[1])
            filter_tuple = tuple(filter_tuple)

            if filter_tuple not in counts:
                print(f"Adding missing filter {filter_tuple} to dictionaries")
                counts[filter_tuple] = 0
                values[filter_tuple] = 0.0
                variances[filter_tuple] = 0.0

            if mssim_score < truncation * original_mssim:
                print(f"Truncated at filter {i}/{len(filters)}")
                break

            remove_filters(model_copy, [filter])
            new_mssim = np.mean([calculate_ms_ssim(model_copy, f'{img_dir}/input_id/simu{i:05d}.npy',
                                                f'{img_dir}/target_from_raw/simu{i:05d}.npy')
                                 for i in range(2, 6)]) # Cambiar por la métrica que se esté usando
            marginal_contribution = mssim_score - new_mssim

            counts[filter_tuple] += 1
            old_avg = values[filter_tuple]
            values[filter_tuple] += (marginal_contribution - values[filter_tuple]) / counts[filter_tuple] # Shapley value
            variances[filter_tuple] += (marginal_contribution - old_avg) * (marginal_contribution - values[filter_tuple])

            mssim_score = new_mssim

        # Bernstein confidence bounds
        for filter in filters:
            filter_tuple = tuple(filter)
            if counts[filter_tuple] > 1:
                std_dev = math.sqrt(variances[filter_tuple] / (counts[filter_tuple] - 1))
                cb = math.sqrt(2 * math.log(2 / delta) / counts[filter_tuple]) * std_dev + \
                     7 * math.log(2 / delta) / (3 * (counts[filter_tuple] - 1))
                upper_bound = values[filter_tuple] + cb
                lower_bound = values[filter_tuple] - cb

                k_largest_value = sorted(values.values(), reverse=True)[min(num_filters - 1, len(values) - 1)]
                if lower_bound > k_largest_value + epsilon or upper_bound < k_largest_value - epsilon:
                    counts[filter_tuple] = float('inf')

        active_filters = [f for f in filters if counts[tuple(f)] < float('inf')]
        print(f"Active filters: {len(active_filters)}")
        
        if len(active_filters) <= num_filters:
            print("Breaking early due to convergence")
            break

        # Save checkpoint every checkpoint_interval samples
        if (sample + 1) % checkpoint_interval == 0:
            ongoing_time = time.time()
            state = {
                'filters': filters,
                'values': {str(k): v for k, v in values.items()},
                'counts': {str(k): v for k, v in counts.items()},
                'variances': {str(k): v for k, v in variances.items()},
                'current_sample': sample + 1,
                'original_mssim': original_mssim, # Cambiar por la métrica que se esté usando
                'active_filters': len(active_filters),
                'ongoing_time': f"{ongoing_time:.2f} seconds"
            }
            save_checkpoint(state, sample + 1)

    return {filter: value for filter, value in values.items()}

def main():
    # Load model
    model = SimpleBeamformer().to(device)
    model.load_state_dict(torch.load(f'{MODEL_PATH}/simple_mse_weights.pth', map_location=device))
    model.eval()

    # Configure parameters  
    num_filters = 20 # Numero de filtros mas importantes a seleccionar
    num_samples = 50000 # Numero de iteraciones maximas
    truncation = 0.25 # Indice de truncamiento temprano
    delta = 0.1 # Valor por defecto para calculo de confidence bounds
    epsilon = 0.0001 # Valor de tolerancia para seleccion de filtros importantes
    checkpoint_interval = 200 # Intervalo de guardado de checkpoints en .json

    # Run TMAB Shapley
    start_time = time.time()
    shapley_values = tmab_shapley(
        model, 
        num_filters,
        num_samples, 
        truncation,
        delta,
        epsilon,
        IMG_DIR,
        checkpoint_interval,
        resume_from_checkpoint=False
    )
    end_time = time.time()
    execution_time = end_time - start_time

    # Print results
    for filter, value in sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)[:num_filters]:
        print(f"Filter {filter}: Shapley value = {value}")

    print(f"Execution time: {execution_time:.2f} seconds")
    
    # # Save final results in .h5
    # with h5py.File(f'{MODEL_PATH}/tmab_shapley_results.h5', 'w') as f:
    #     shapley_group = f.create_group('shapley_values')
    #     for filter, value in shapley_values.items():
    #         shapley_group.create_dataset(f"{filter[0]}_{filter[1]}", data=value)
    #     f.create_dataset('execution_time', data=execution_time)

if __name__ == "__main__":
    main()
