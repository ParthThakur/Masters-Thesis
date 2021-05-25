import numpy as np
import h5py
import time
from datetime import datetime
from tqdm import tqdm

from helpers.Simulator import FRB
from helpers.MakeExamplePlot import make_plot

# Development seed: 260795. Uncomment to get data generated during development.
# np.random.seed(260795)

SNR_MIN = 4
SNR_MAX = 10  # Minimum and maximum signal to noise ratio.
N_SIM = 5000  # Number of simulations.
SHAPE = (128, 512)  # Shape of the array: (frequency resolution, time resolution) Higher numbers = Higher resolution
fn_out = f'data/{datetime.now().strftime("%y%m%d-%H%M%S")}.h5'  # Output file for simulations.


def make_frb_simulations(shape=SHAPE, n_sim=N_SIM):
    FRBS_ = [FRB(shape=shape) for _ in range(n_sim)]
    print(f"Simulating {n_sim} FRB.")
    [event.simulateFRB(SNRmin=SNR_MIN, SNRmax=SNR_MAX) for event in tqdm(FRBS_)]
    return np.array([F.simulatedFRB for F in FRBS_]), np.array([F.SNR for F in FRBS_])


def make_backgrounds(shape=SHAPE, n_sim=N_SIM):
    print("Generating background.")
    backgrounds = [np.random.standard_normal(shape) for _ in tqdm(range(n_sim))]
    return np.array(backgrounds)


# Make an example image for simulations
# make_plot(SHAPE)

with h5py.File(fn_out, 'w') as fw:
    t1 = time.time()
    FRBS, SNRS = make_frb_simulations()
    BAK = make_backgrounds()
    print(f'File name: {fn_out}')
    fw.create_dataset('FRB', data=FRBS)
    fw.create_dataset('BAK', data=BAK)
    fw.create_dataset('parameters', data=SNRS)
    print(f'Saved file with {N_SIM} FRBs with size {FRBS.nbytes * 2 / 1024 ** 3}GB. \n'
          f'Process completed in {time.strftime("%H:%M:%S", time.gmtime(time.time() - t1))}.')
