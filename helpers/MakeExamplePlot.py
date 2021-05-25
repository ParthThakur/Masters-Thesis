import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .Simulator import FRB


def make_plot(SHAPE):
    # create simulation objects and give an FRB to each of them
    simulated_events = [FRB(shape=SHAPE) for _ in range(8)]
    for event in simulated_events:
        event.simulateFRB(SNRmin=6, SNRmax=20)

    # plot the simulated events
    fig_simulated, ax_simulated = plt.subplots(nrows=4, ncols=2, figsize=(18, 14))

    for axis, event in zip(ax_simulated.flatten(), simulated_events):
        im = axis.imshow(event.simulatedFRB, extent=[0, event.nt, event.frequencies[0], event.frequencies[-1]],
                         origin='lower', aspect='auto')
        axis.set(title=f"SNR: {np.round(event.SNR, 2)}", xlabel='time (ms)', ylabel='frequency (MHz)')
        axis.set_yticks(np.arange(event.frequencies[0], event.frequencies[-1], 350))

    # make a colorbar
    cbar_ax = fig_simulated.add_axes([0.83, 0.05, 0.02, 0.9])
    fig_simulated.colorbar(im, cax=cbar_ax)

    fig_simulated.tight_layout()
    fig_simulated.subplots_adjust(right=0.8, wspace=0.3)
    plt.savefig(f'media/FRB and RFI plot - {datetime.now().strftime("%yy%mm%dd-%H%M%S")}.png')
    plt.show()
