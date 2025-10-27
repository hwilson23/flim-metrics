import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from scipy.signal import convolve
    import matplotlib.pyplot as plt

    def simulate_multi_exponential_decay_with_irf(lifetimes, amplitudes, irf_fwhm, time_bins=256, time_range=(0, 12.5), num_photons=10000, background=0, rising_factor=1.0):
        """
        Simulate a multi-exponential fluorescence lifetime decay curve with a realistic IRF and rising component.
    
        Parameters:
        -----------
        lifetimes : list or array
            List of fluorescence lifetimes (in nanoseconds)
        amplitudes : list or array
            Relative amplitudes for each lifetime component (will be normalized)
        irf_fwhm : float
            Full Width at Half Maximum of the IRF (in nanoseconds)
        time_bins : int
            Number of time bins to use
        time_range : tuple
            (min_time, max_time) in nanoseconds
        num_photons : int
            Total number of photons to simulate
        background : float
            Expected background counts per bin
        rising_factor : float
            Controls the steepness of the rising edge. Higher values create a sharper rise
        
        Returns:
        --------
        times : numpy array
            Time points (center of bins) of the simulation
        decay_curve : numpy array
            Simulated decay curve including IRF and rising component
        irf : numpy array
            The generated IRF (Gaussian)
        pure_decay : numpy array
            The pure exponential decay before convolution
        """
        min_time, max_time = time_range
        if len(lifetimes) != len(amplitudes):
            raise ValueError('The number of lifetimes must match the number of amplitudes')
        norm_amplitudes = np.array(amplitudes) / np.sum(amplitudes)
        times = np.linspace(min_time, max_time, time_bins)
        dt = times[1] - times[0]
        sigma = irf_fwhm / 2.35482
        center_position = min_time + 3 * sigma
        _irf = np.exp(-0.5 * ((times - center_position) / sigma) ** 2)
        _irf = _irf / np.sum(_irf)
        pure_decay = np.zeros_like(times)
        decay_start_idx = np.argmin(np.abs(times - center_position))
        decay_times = times[decay_start_idx:] - times[decay_start_idx]
        rise_time = min(lifetimes) / rising_factor
        rising_component = 1 - np.exp(-decay_times / rise_time)
        multi_exp_decay = np.zeros_like(decay_times)
        for i, (lifetime, amplitude) in enumerate(zip(lifetimes, norm_amplitudes)):
            decay_component = amplitude * np.exp(-decay_times / lifetime)
            multi_exp_decay += decay_component
        modeled_curve = rising_component * multi_exp_decay
        pure_decay[decay_start_idx:] = modeled_curve
        if np.sum(pure_decay) > 0:
            pure_decay = pure_decay * (num_photons / np.sum(pure_decay))
        decay_curve = convolve(pure_decay, _irf, mode='same')
        decay_curve = np.random.poisson(decay_curve)
        if background > 0:
            bg_counts = np.random.poisson(background, size=time_bins)
            decay_curve = decay_curve + bg_counts
        return (times, decay_curve, _irf, pure_decay)

    def plot_multi_exponential_decay_enhanced(times, decay_curve, irf, pure_decay, lifetimes, amplitudes, irf_fwhm):
        """
        Create enhanced plots for visualizing the multi-exponential decay curve, IRF, and pure decay separately.
    
        Parameters:
        -----------
        times : numpy array
            Time points
        decay_curve : numpy array
            Simulated decay curve including IRF and photon noise
        irf : numpy array
            The generated IRF
        pure_decay : numpy array
            The pure model before convolution
        lifetimes : list or array
            The lifetimes used for simulation
        amplitudes : list or array
            The amplitudes used for simulation
        irf_fwhm : float
            FWHM of the IRF used
        
        Returns:
        --------
        figs : list
            List of figure objects
        """
        figs = []
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(times, _irf, 'r-', linewidth=2)
        ax1.fill_between(times, 0, _irf, color='red', alpha=0.3)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Normalized Intensity')
        ax1.set_title(f'Instrument Response Function (FWHM = {irf_fwhm:.2f} ns)')
        max_irf = np.max(_irf)
        half_max = max_irf / 2
        above_half_max = _irf >= half_max
        if np.any(above_half_max):
            indices = np.where(above_half_max)[0]
            left_idx = indices[0]
            right_idx = indices[-1]
            left_x = times[left_idx]
            right_x = times[right_idx]
            ax1.axhline(y=half_max, color='k', linestyle='--', alpha=0.5)
            ax1.axvline(x=left_x, color='k', linestyle='--', alpha=0.5)
            ax1.axvline(x=right_x, color='k', linestyle='--', alpha=0.5)
            ax1.annotate(f'FWHM = {irf_fwhm:.2f} ns', xy=((left_x + right_x) / 2, half_max * 1.1), xytext=((left_x + right_x) / 2, half_max * 1.5), arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), ha='center')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        figs.append(fig1)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(times, decay_curve, 'b.', alpha=0.5, label='Measured Decay (IRF Convolved)')
        ax2.plot(times, pure_decay, 'g-', linewidth=2, alpha=0.7, label='Pure Model')
        title = f'Multi-exponential Decay: '
        for i, (tau, amp) in enumerate(zip(lifetimes, amplitudes)):
            if i > 0:
                title += ', '
            norm_amp = amp / sum(amplitudes)
            title += f'τ{i + 1}={tau:.2f}ns ({norm_amp:.2f})'
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Counts')
        ax2.set_title(title)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        figs.append(fig2)
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        irf_max = np.max(decay_curve) * 0.8
        irf_scaled = _irf * (irf_max / np.max(_irf)) if np.max(_irf) > 0 else _irf
        ax3a.plot(times, decay_curve, 'b.', alpha=0.5, label='Measured Decay')
        ax3a.plot(times, pure_decay, 'g-', linewidth=2, alpha=0.7, label='Pure Model')
        ax3a.plot(times, irf_scaled, 'r-', linewidth=2, alpha=0.7, label='IRF (scaled)')
        ax3a.set_xlabel('Time (ns)')
        ax3a.set_ylabel('Counts')
        ax3a.set_title('Linear Scale View')
        ax3a.legend()
        ax3a.grid(True, alpha=0.3)
        zoom_end_idx = int(len(times) * 0.2)
        zoom_end_time = times[zoom_end_idx]
        peak_idx = np.argmax(decay_curve[:zoom_end_idx * 2])
        peak_value = decay_curve[peak_idx]
        ax3b.plot(times, decay_curve, 'b.', alpha=0.5, label='Measured Decay')
        ax3b.plot(times, pure_decay, 'g-', linewidth=2, alpha=0.7, label='Pure Model')
        ax3b.plot(times, irf_scaled, 'r-', linewidth=2, alpha=0.7, label='IRF (scaled)')
        ax3b.set_xlim(0, zoom_end_time * 1.5)
        ax3b.set_ylim(0, peak_value * 1.1)
        ax3b.set_xlabel('Time (ns)')
        ax3b.set_ylabel('Counts')
        ax3b.set_title('Zoom on Rising Edge')
        ax3b.legend()
        ax3b.grid(True, alpha=0.3)
        plt.tight_layout()
        figs.append(fig3)
        return figs

    def calculate_average_lifetime(lifetimes, amplitudes):
        """
        Calculate amplitude-weighted average lifetime for a multi-exponential decay.
    
        Parameters:
        -----------
        lifetimes : list or array
            List of lifetimes (in nanoseconds)
        amplitudes : list or array
            List of corresponding amplitudes
        
        Returns:
        --------
        avg_lifetime : float
            Amplitude-weighted average lifetime
        """
        norm_amplitudes = np.array(amplitudes) / np.sum(amplitudes)
        avg_lifetime = np.sum(np.array(lifetimes) * norm_amplitudes)
        return avg_lifetime
    if __name__ == '__main__':
        lifetimes = [1.5, 4.0]
        amplitudes = [0.7, 0.3]
        irf_fwhm = 0.5
        time_bins = 256
        time_range = (0, 12.5)
        num_photons = 10000
        background_level = 5
        rising_factor = 2.0
        times, decay_curve, _irf, pure_decay = simulate_multi_exponential_decay_with_irf(lifetimes=lifetimes, amplitudes=amplitudes, irf_fwhm=irf_fwhm, time_bins=time_bins, time_range=time_range, num_photons=num_photons, background=background_level, rising_factor=rising_factor)
        figs = plot_multi_exponential_decay_enhanced(times=times, decay_curve=decay_curve, irf=_irf, pure_decay=pure_decay, lifetimes=lifetimes, amplitudes=amplitudes, irf_fwhm=irf_fwhm)
        plt.show()
        avg_lifetime = calculate_average_lifetime(lifetimes, amplitudes)
        total_photons = np.sum(decay_curve)
        print(f'Lifetimes: {lifetimes}')
        print(f'Amplitudes: {amplitudes}')
        print(f'Amplitude-weighted average lifetime: {avg_lifetime:.2f} ns')
        print(f'IRF FWHM: {irf_fwhm} ns')
        print(f'Total photons detected: {total_photons}')
    return decay_curve, np, plt, pure_decay, times


@app.cell
def _(decay_curve, np, plt, pure_decay, times):
    from scipy import signal

    def add_rising_edge(time, existing_decay, irf):
        """
        Adds a realistic rising edge to an existing exponential decay model
        by convolving it with an instrument response function (IRF).
    
        Parameters:
        -----------
        time : numpy.ndarray
            Time axis array (in nanoseconds or other consistent units)
        existing_decay : numpy.ndarray
            Existing exponential decay model with the same length as time
        irf : numpy.ndarray
            Instrument response function with the same length as time
    
        Returns:
        --------
        numpy.ndarray
            Modified decay curve with realistic rising edge
        """
        time = np.asarray(time)
        existing_decay = np.asarray(existing_decay)
        _irf = np.asarray(_irf)
        irf_normalized = _irf / np.sum(_irf)
        convolved_signal = signal.convolve(existing_decay, irf_normalized, mode='same')
        return convolved_signal

    def shift_decay_for_convolution(time, existing_decay, shift_amount):
        """
        Shifts an existing decay model to ensure proper alignment with the IRF
        before convolution. This is useful when the original decay starts at t=0
        but needs to be shifted to align with the IRF peak.
    
        Parameters:
        -----------
        time : numpy.ndarray
            Time axis array
        existing_decay : numpy.ndarray
            Existing exponential decay model
        shift_amount : float
            Amount to shift the decay (positive shifts right, negative shifts left)
    
        Returns:
        --------
        numpy.ndarray
            Shifted decay model
        """
        from scipy.interpolate import interp1d
        f = interp1d(time, existing_decay, bounds_error=False, fill_value=0)
        shifted_time = time - shift_amount
        shifted_decay = f(shifted_time)
        return shifted_decay

    def align_and_add_rising_edge(time, existing_decay, irf, t0=None):
        """
        Aligns the decay model with the IRF, then adds the rising edge.
    
        Parameters:
        -----------
        time : numpy.ndarray
            Time axis array
        existing_decay : numpy.ndarray
            Existing exponential decay model
        irf : numpy.ndarray
            Instrument response function
        t0 : float, optional
            Time zero position. If None, uses the peak of the IRF
    
        Returns:
        --------
        numpy.ndarray
            Properly aligned decay model with rising edge
        """
        if t0 is None:
            irf_peak_idx = np.argmax(_irf)
            t0 = time[irf_peak_idx]
        decay_start_idx = np.where(existing_decay > 0.001 * np.max(existing_decay))[0][0]
        decay_start_time = time[decay_start_idx]
        shift_amount = decay_start_time - t0
        if abs(shift_amount) > 1e-10:
            shifted_decay = shift_decay_for_convolution(time, existing_decay, shift_amount)
        else:
            shifted_decay = existing_decay
        decay_with_rising_edge = add_rising_edge(time, shifted_decay, _irf)
        return decay_with_rising_edge
    time = np.linspace(-5, 20, 5000)
    tau = 4.0
    ideal_decay = np.zeros_like(time)
    ideal_decay[time >= 0] = np.exp(-time[time >= 0] / tau)
    irf_width = 0.25
    irf_center = 6.25
    _irf = np.exp(-(times - irf_center) ** 2 / (2 * irf_width ** 2))
    realistic_decay = add_rising_edge(times, pure_decay, _irf)
    fulltime = []
    plt.plot(times, _irf * np.max(pure_decay), 'g-', label='IRF (normalized, multiplied)')
    plt.plot(time, ideal_decay * np.max(pure_decay), 'b--', label='Instant (norm multiplied)')
    plt.plot(times, realistic_decay, 'r.', label='With rising edge')
    plt.plot(times, pure_decay, label='pure decay')
    plt.plot(times, decay_curve, 'b.', alpha=0.5, label='Measured Photons')
    plt.xlim([-5, 15])
    plt.xlabel('Time (ns)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Demonstration of Rising Edge in Lifetime Decay')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.grid(True)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(times, _irf, 'r-', linewidth=2)
    ax1.fill_between(times, 0, _irf, color='red', alpha=0.3)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Normalized Intensity')
    max_irf = np.max(_irf)
    half_max = max_irf / 2
    above_half_max = _irf >= half_max
    if np.any(above_half_max):
        indices = np.where(above_half_max)[0]
        left_idx = indices[0]
        right_idx = indices[-1]
        left_x = times[left_idx]
        right_x = times[right_idx]
        ax1.axhline(y=half_max, color='k', linestyle='--', alpha=0.5)
        ax1.axvline(x=left_x, color='k', linestyle='--', alpha=0.5)
        ax1.axvline(x=right_x, color='k', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    return


if __name__ == "__main__":
    app.run()
