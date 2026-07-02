import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt



def constrained_bandpass(filt_b1, filt_band, fs = 500.0, min_freq = 1.0, min_band = 2.0, cutoff = 50, freq_high = None):

    # upper bound: freq_high if given, else cutoff
    hi = float(cutoff if freq_high is None else freq_high)

    filt_beg_freq = torch.clamp(
                                torch.abs(filt_b1) + min_freq / fs,
                                min_freq / fs,
                                (hi - min_band) / fs
                            )

    filt_end_freq = torch.clamp(
                                filt_beg_freq + (torch.abs(filt_band) + min_band / fs),
                                (min_freq + min_band) / fs,
                                hi / fs)

    return filt_beg_freq * fs, filt_end_freq * fs


# sinc function.
def sinc(band,t_right, device):

    # all of these ingerit .to(device) from band and t_right!
    y_right= torch.sin(2*torch.pi*band*t_right)/(2*torch.pi*band*t_right) 

    y_left= torch.flip(y_right, [0])

    # adding the 1 due to sinc(x) being undefined numerically when t = 0.
    y=torch.cat([y_left, torch.ones(1, device = device) ,y_right])

    return y




class SincConvolution(nn.Module):
    def __init__(self, N_filt, Filt_dim, fs, cutoff = 50, device = None,
                 freq_low = 1.0, freq_high = None, min_band = 2.0,
                 fixed_bands = None):
        super(SincConvolution, self).__init__()
        
        # fixed_bands : optional list of (low_hz, high_hz) tuples. When provided,
        #   the filterbank is FROZEN to these exact bands (not trained), N_filt is
        #   set to len(fixed_bands), and gradient descent cannot move the filters.
        #   This guarantees full, user-specified spectral coverage.
        self.fixed_bands = fixed_bands
        if fixed_bands is not None:
            N_filt = len(fixed_bands)

        # number of filters.
        self.N_filt = N_filt
        # the length of the filters. better if odd for a true center sample at t = 0.
        self.Filt_dim = Filt_dim
        # the sampling frequency
        self.fs = fs 

        # if the data is already preprocessed and bandpassed to a cutoff point.
        self.cutoff = cutoff

        # user-specified hard frequency constraints for the trainable filters.
        # freq_low  : minimum allowed low-cutoff (Hz). Filters cannot train below this.
        # freq_high : maximum allowed high-cutoff (Hz). Defaults to `cutoff` if None.
        # min_band  : minimum filter bandwidth (Hz).
        # These are hard constraints applied every forward pass via clamping.
        self.freq_low  = float(freq_low)
        self.freq_high = float(cutoff if freq_high is None else freq_high)
        self.min_band  = float(min_band)
        if fixed_bands is None:
            assert self.freq_high <= cutoff, (
                f"freq_high={self.freq_high} must be <= cutoff={cutoff}")
            assert self.freq_low + self.min_band <= self.freq_high, (
                f"freq_low ({self.freq_low}) + min_band ({self.min_band}) must be "
                f"<= freq_high ({self.freq_high})")
        
        # nyquist cutoff frequency: highest frequency in a signal that can be accurately captured and reconstructed
        # self.cutoff = self.fs/2
        self.freq_scale = fs * 1.0


        # initialise filter anchors within the user-specified frequency band
        low_freq  = self.freq_low
        high_freq = self.freq_high

        # uniform initialization.
        # freq_init = np.random.uniform(low_freq,high_freq, N_filt)

        # try a different initialization based on log spaced centers.
        # freq_init = np.geomspace(low_freq, high_freq, N_filt)

        # try a different initialization based on a constant 2 Hz.
        # freq_init = np.full(N_filt, 20)
        
        # linear spaced initialization.
        # freq_init = np.linspace(low_freq, high_freq, N_filt)
        
        # band_init = 2
        

        # initialized value for the unconstrained parameter for the low-cutoff frequences.
        # b1 = freq_init
        # initialized value for the raw unconstrained parameter for the filter bandwidth
        # b2 = np.zeros_like(b1) + band_init 


        if self.fixed_bands is not None:
            # FROZEN filterbank: use the exact user-specified bands.
            bands = np.asarray(self.fixed_bands, dtype=float)   # (N_filt, 2)
            assert bands.shape[1] == 2, "fixed_bands must be a list of (low, high) tuples"
            b1 = bands[:, 0]                     # low cutoffs
            b2 = bands[:, 1]                     # high cutoffs
            assert np.all(b2 > b1), "each fixed band must have high > low"
            # parameters exist but are NOT trained (requires_grad=False), so the
            # forward pass reconstructs exactly these bands every time.
            self.filt_b1   = nn.Parameter(torch.from_numpy(b1 / self.freq_scale),
                                          requires_grad=False)
            self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale),
                                          requires_grad=False)
        else:
            # LEARNABLE filterbank: log-spaced overlapping initialisation (Mel-style).
            # N_filt + 2 anchors; filter i spans [anchors[i], anchors[i+2]].
            anchors = np.geomspace(low_freq, high_freq, self.N_filt + 2)
            b1 = anchors[:-2]
            b2 = anchors[2:]
            self.filt_b1   = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
            self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))

        
        # controls where the data lives. so if your model is in cuda, so does your input data.
        self.device = device 
        
    def __format(self, x):
        # Format should be (B, C, T)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(self.device)

        else:
            x = x.to(self.device)

        return x
    
    def forward(self, x, test_plot=False):
        # derive the device from the module's own parameters so that a model
        # moved with .to(device) works even though self.device is a stale
        # plain attribute (.to moves parameters/buffers, not python attributes).
        dev = self.filt_b1.device
        self.device = dev

        # ensure x is on the same device.
        x = x.to(dev) if isinstance(x, torch.Tensor) else torch.from_numpy(x).float().to(dev)

        # create N filters with a length of Filt_dim.
        filters = torch.zeros((self.N_filt, self.Filt_dim), device = dev)
        # filters_list = []

        

        # sanity check: look at how the band passes look
        # ------------------------------------------------------------------------------
        band_passes = torch.zeros((self.N_filt, self.Filt_dim), device = dev)
        # ------------------------------------------------------------------------------

        
        # the length of the bandpass filter 
        N = self.Filt_dim

        # time (seconds) for the right symmetric side of the signal. 
        t_right = torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2), device = dev) / self.fs


        # hard constraints from user-specified bounds (Hz, normalised by freq_scale)
        min_freq = self.freq_low
        min_band = self.min_band
        max_freq = self.freq_high

        # The constrained/clamped versions of the learned parameters, basically the real low and high-cutoff of the bandpass.
        # will be of shape N_filt for N filters.
        # filt_beg_freq = torch.clamp(
        #                             # force the low-cutoff to stay positive, and add min_freq to guarantee at least min_freq.
        #                             torch.abs(self.filt_b1) + min_freq / self.freq_scale,
        #                             # want to clamp values below min_freq to min_freq.
        #                             min_freq / self.freq_scale,
        #                             # want to clamp values that are above (cutoff - min_band). 
        #                             ((self.cutoff) - int(min_band)) / self.freq_scale
        #                         )


        # will be of shape N_filt for N filters.
        # filt_end_freq = torch.clamp(
        #                             # add the band to the filter's beginning frequencies. add min_band to guarantee at least min_band.
        #                             filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale),
        #                             # want to clamp high-cutoff values to be at least min_freq + min_band.
        #                             int(min_freq + min_band) / self.freq_scale, 
        #                             # want to clamp high-cutoff values to be at most the cutoff frequency.
        #                             (self.cutoff) / self.freq_scale)


        # low cutoff: at least freq_low, at most freq_high - min_band
        if self.fixed_bands is not None:
            # FROZEN bands: use the stored parameters exactly, no offset/clamp.
            # (filt_b1 holds low cutoff, filt_band holds bandwidth, both /freq_scale)
            filt_beg_freq = self.filt_b1
            filt_end_freq = self.filt_b1 + self.filt_band
        else:
            filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
            filt_beg_freq = torch.clamp(
                filt_beg_freq,
                min=min_freq / self.freq_scale,
                max=(max_freq - min_band) / self.freq_scale,
            )
            # bandwidth: at least min_band, and not pushing high cutoff past freq_high
            filt_band = torch.abs(self.filt_band) + min_band / self.freq_scale
            max_band  = max_freq / self.freq_scale - filt_beg_freq
            filt_band = torch.minimum(filt_band, max_band)
            filt_end_freq = filt_beg_freq + filt_band

        # why add min min_freq and min_band and just rely on the clamp? perhaps because if it gets clamped to exactly 
            # min_freq or min_band and gradients become zero and it can't learn to move out the boundary?


        # they will both be of shape N_filt.
        # print(filt_beg_freq.shape, filt_end_freq.shape)


        n = torch.arange(N, device = dev) # 0,..., N
        
        # filter window (hamming)
        window=0.54-0.46*torch.cos(2*torch.pi*n/N);

        for i in range(self.N_filt): # loop through N filters.
            # print(filt_beg_freq[i] * self.freq_scale, filt_end_freq[i] * self.freq_scale)
            
            # rescale to actual frequency inside sinc function.
            # low-pass filter of the lower frequency 
            low_pass1 = 2 * filt_beg_freq[i] * sinc(filt_beg_freq[i] * self.freq_scale, t_right, device=dev)
            # low-pass filter of the higher frequency
            low_pass2 = 2 * filt_end_freq[i] * sinc(filt_end_freq[i] * self.freq_scale, t_right, device=dev)
            
            # lower-cutoff low pass filter from a higher low pass flter forms a bandpass filter.
            band_pass = (low_pass2 - low_pass1)
            band_pass = band_pass / torch.max(band_pass)   #normalize to one
            
            band_passes[i, :] = band_pass
        
            # element-wise multplication with hamming window.
            filters[i, :] = band_pass * window



        # sanity check: plot just to check the filters.
        # ------------------------------------------------------------------------------
        if test_plot:


            self.filt_b2 = (self.filt_b1.detach().numpy() + self.filt_band.detach().numpy())
            plt.figure()
            for i, (low, high) in enumerate(zip(self.filt_b1.detach().numpy(), self.filt_b2)):
                # print(low * self.freq_scale , high * self.freq_scale)
                plt.plot([low * self.freq_scale , high * self.freq_scale] , [i, i], "b--")
            plt.title("Initialized Filters")
            plt.xlabel("Frequency")
            plt.show()
        
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 3, 1)
            plt.plot(band_passes.detach().numpy().T)
            plt.title("band passes")
            
            plt.subplot(1, 3, 2)
            plt.plot(window.detach().cpu().numpy())
            plt.title("hamming window")
            
            plt.subplot(1, 3, 3)
            plt.plot(filters.detach().cpu().numpy().T)
            plt.title("SincNet band-pass filter")
            plt.show()
        # ------------------------------------------------------------------------------



        # x: (B, C, T) -> (B, 1, C, T) basically treat it as it it were a grayscale image with one feature map as we convolve it.
        x = x.unsqueeze(dim = 1) # the index at which to insert the singleton dimension.

        # filters are stored in (num_filters, Filt_dim) -> kernel (num_filters, 1, 1, Filt_dim) 
        # in conv 2D, kernels take in (out channels, in channels / feature maps, kernel height, kernel width)
        kernel = filters.unsqueeze(1).unsqueeze(1)
        
        # out (valid convolution) = (Batch, N_filt, C, T - Filt_dim + 1)
        # NO POINTWISE CONVOLUTION! There is only one feature map at the start so no channel mixing.
        # our kernels are NOT free parameters like usual conv2D operations, we HAVE our values for the filters already,
            # we are just using conv2d to actually do the operation.
        out = F.conv2d(x, kernel)
        
        # print(filters.requires_grad)  

        return out, filters, (filt_beg_freq, filt_end_freq)

