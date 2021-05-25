"""
Package to simulate fast radio bursts and radio frequency interference.
The parameters used in this package are based on the maths provided by Petroff et. al.
Their paper is available at: https://doi.org/10.1007/s00159-019-0116-6

Author: Parth Thakur.
Source: DominicL3.

"""
import numpy as np
from scipy.signal import fftconvolve


class FRB(object):
    """ Class to generate a realistic fast radio burst and
    add the event to data, including scintillation and
    temporal scattering. @source liamconnor
    """

    def __init__(self, shape=(64, 256), f_ref=1350, bandwidth=1500, max_width=4, tau=0.1):
        assert type(shape) == tuple and len(shape) == 2, "shape needs to be a tuple of 2 integers"
        self.shape = shape

        # reference frequency (MHz) of observations
        self.f_ref = f_ref

        # maximum width of pulse, high point of uniform distribution for pulse width
        self.max_width = max_width

        # number of bins/data points on the time (x) axis
        self.nt = shape[1]

        # frequency range for the pulse, given the number of channels
        self.frequencies = np.linspace(f_ref - bandwidth // 2, f_ref + bandwidth // 2, shape[0])

        # where the pulse will be centered on the time (x) axis
        self.t0 = np.random.randint(-shape[1] + max_width, shape[1] - max_width)

        # scattering timescale (milliseconds)
        self.tau = tau

        # randomly generated SNR and FRB generated after calling injectFRB()
        self.SNR = None
        self.FRB = None
        self.dedispersed_FRB = None

        '''Simulates background noise similar to the .ar 
        files. Backgrounds will be injected with FRBs to 
        be used in classification later on.'''
        self.background = np.random.standard_normal(self.shape)
        self.simulatedFRB = None

    def dispersed_profile(self, g, disp_ind):
        """Model a pulse with a dispersed profile.
        The parameters used in this function were finalised by empirical analysis based on anecdotal evidence on the
        shape of an FRB."""
        self.dedispersed_FRB = g
        for index, roll in enumerate(np.geomspace(200, 300, num=self.shape[0])):
            roll_ = roll / 100 - 10
            g[index] = np.roll(g[index], int(roll_ * disp_ind))
        return g[::-1]

    def gaussian_profile(self):
        """Model pulse as a normalized Gaussian."""
        t = np.linspace(-self.nt // 2, self.nt // 2, self.nt)
        g = np.exp(-(t / np.random.randint(1, self.max_width)) ** 2)

        if not np.all(g > 0):
            g += 1e-18

        # clone Gaussian into 2D array with NFREQ rows
        g = np.tile(g, (self.shape[0], 1))
        g = self.dispersed_profile(g, np.random.randint(50, 75))

        return g

    def scatter_profile(self):
        """ Include exponential scattering profile."""
        tau_nu = self.tau * (self.frequencies / self.f_ref) ** -4
        t = np.linspace(0, self.nt // 2, self.nt)

        prof = np.exp(-t / tau_nu.reshape(-1, 1)) / tau_nu.reshape(-1, 1)
        return prof / np.max(prof, axis=1).reshape(-1, 1)

    def pulse_profile(self):
        """ Convolve the gaussian and scattering profiles
        for final pulse shape at each frequency channel.
        """
        gaus_prof = self.gaussian_profile()
        scat_prof = self.scatter_profile()

        # convolve the two profiles for each frequency
        pulse_prof = np.array([fftconvolve(gaus_prof[i], scat_prof[i])[:self.nt] for i in np.arange(self.shape[0])])
        # normalize! high frequencies should have narrower pulses
        # noinspection PyUnresolvedReferences
        pulse_prof /= np.trapz(pulse_prof, axis=1).reshape(-1, 1)
        return pulse_prof

    def scintillate(self):
        """Approximate frequency scintillation as the positive half of
        a cosine function. Randomize the phase and decorrelation bandwidth.
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()

        # Make number of scintils between 0 and 10
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(7)))

        # set number of scintillations to 0 if it's below 1
        if nscint < 1:
            nscint = 0

        # make envelope a cosine function
        envelope = np.cos(2 * np.pi * nscint * (self.frequencies / self.f_ref) ** -2 + scint_phi)

        # set all negative elements to zero and add small factor
        envelope[envelope < 0] = 0
        envelope += 0.1

        # add scintillation to pulse profile
        pulse = self.pulse_profile()
        pulse *= envelope.reshape(-1, 1)
        self.FRB = pulse

    def roll(self):
        """Move FRB to random location of the time axis (in-place),
        ensuring that the shift does not cause one end of the FRB
        to end up on the other side of the array."""
        bin_shift = np.random.randint(low=-self.shape[1] // 2 + self.shape[1] // 4,
                                      high=self.shape[1] // 2 - self.shape[1] // 4)
        self.FRB = np.roll(self.FRB, bin_shift, axis=1)

    def fractional_bandwidth(self, frac_low=0.5, frac_high=0.9):
        """Cut some fraction of the full pulse out."""
        # Fraction of frequency (y) axis for the signal
        frac = np.random.uniform(frac_low, frac_high)
        nchan = self.shape[0]

        # collect random fraction of FRB and add to background
        stch = np.random.randint(0, nchan * (1 - frac))
        slice_freq = slice(stch, int(stch + (nchan * frac)))
        slice_FRB = np.copy(self.FRB[slice_freq])
        self.FRB[:, :] = 1e-18
        self.FRB[slice_freq] = slice_FRB

    def sample_SNR(self, SNRmin=8, SNR_sigma=1.0, SNRmax=30):
        """Sample peak SNR from log-normal distribution and throw
        out any value greater than SNRmax."""
        if SNRmin < 0:
            raise ValueError('Minimum SNR cannot be negative')
        if SNRmin > SNRmax:
            raise ValueError('SNRmin cannot be greater than SNRmax')

        random_SNR = SNRmin + np.random.lognormal(mean=1.0, sigma=SNR_sigma)
        if random_SNR < SNRmax:
            self.SNR = random_SNR
            return random_SNR
        else:
            return self.sample_SNR(SNRmin, SNR_sigma, SNRmax)

    def injectFRB(self, SNR, background=None):
        """Inject the FRB into freq-time array of Gaussian noise"""
        if background is None:
            background = self.background

        # get 1D noise and multiply signal by given SNR
        noise_profile = np.mean(background, axis=0)
        peak_value = SNR * np.std(noise_profile)
        profile_FRB = np.mean(self.dedispersed_FRB, axis=0)

        # make a signal with given SNR
        signal = self.FRB * (peak_value / np.max(profile_FRB))
        return signal

    def simulateFRB(self, background=None, SNRmin=8, SNR_sigma=1.0, SNRmax=15):
        """Combine everything together and inject the FRB into a
        background array of Gaussian noise for the simulation. After
        this method works and is detected by the neural network, proceed
        to inject the FRB into the actual noise files given by psrchive."""
        if background is None:
            background = self.background

        # Create the FRB
        self.scintillate()  # make the pulse profile with scintillation
        self.fractional_bandwidth()  # cut out some of the bandwidth
        self.roll()  # move the FRB around freq-time array
        self.sample_SNR(SNRmin, SNR_sigma, SNRmax)  # get random SNR

        # add to the Gaussian noise
        self.simulatedFRB = background + self.injectFRB(background=background, SNR=self.SNR)

"""
Bibliography

DominicL3, FRB FBI: Teaching a Neural Network to Distinguish Between Fast Radio Bursts and Radio Frequency Interference,
available at: https://github.com/DominicL3/hey-aliens/blob/master/simulateFRBclassification/FRBclassifier_notebook.ipynb
accessed on: 08-08-2020

liamconnor, injectfrb,
available at: https://github.com/liamconnor/injectfrb
accessed on: 21-07-2020
"""
