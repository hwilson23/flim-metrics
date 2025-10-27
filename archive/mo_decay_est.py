import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell
def _(export_decay_data_as_3d_tiff):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    def generate_lifetime_decay_photon_simulation(num_photons, lifetime, time_range=(0, 12.5), time_bins=256, background=0, instrument_response=None):
        """
        Simulate a fluorescence lifetime decay curve using direct photon arrival time simulation.
    
        Parameters:
        -----------
        num_photons : int
            Total number of photons to simulate
        lifetime : float
            Fluorescence lifetime (in nanoseconds)
        time_range : tuple
            (min_time, max_time) in nanoseconds
        time_bins : int
            Number of time bins to use
        background : float
            Expected background counts per bin
        instrument_response : array or None
            Instrument response function (probability distribution)
        
        Returns:
        --------
        times : numpy array
            Time points (center of bins) of the simulation
        counts : numpy array
            Photon counts at each time point
        """
        min_time, max_time = _time_range
        photon_arrival_times = np.random.exponential(scale=lifetime, size=_num_photons)
        print(f'photon count arrial ========{photon_arrival_times.shape}')
        photon_arrival_times = photon_arrival_times[photon_arrival_times <= max_time]
        bin_edges = np.linspace(min_time, max_time, _time_bins + 1)
        _counts, _ = np.histogram(photon_arrival_times, bins=bin_edges)
        _times = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if instrument_response is not None:
            if len(instrument_response) != _time_bins:
                irf_resized = np.interp(np.linspace(0, 1, _time_bins), np.linspace(0, 1, len(instrument_response)), instrument_response)
                irf_normalized = irf_resized / np.sum(irf_resized)
            else:
                irf_normalized = instrument_response / np.sum(instrument_response)
            _counts = np.convolve(_counts, irf_normalized, mode='same')
        if background > 0:
            bg_counts = np.random.poisson(background, size=_time_bins)
            _counts = _counts + bg_counts
        return (_times, _counts)

    def exponential_model(t, a, tau, c):
        """
        Single exponential decay model with background.
    
        Parameters:
        t : time
        a : amplitude
        tau : lifetime
        c : background constant
        """
        return a * np.exp(-t / tau) + c

    def fit_lifetime_decay(times, counts, initial_guess=None):
        """
        Fit the lifetime decay curve with a single exponential model.
    
        Parameters:
        -----------
        times : numpy array
            Time points
        counts : numpy array
            Count data
        initial_guess : tuple or None
            Initial parameters (amplitude, lifetime, background)
        
        Returns:
        --------
        params : tuple
            Fitted parameters (amplitude, lifetime, background)
        covariance : array
            Covariance matrix of the fit
        """
        if initial_guess is None:
            background = np.min(_counts)
            amplitude = np.max(_counts) - background
            lifetime = _times[len(_times) // 3] - _times[0]
            initial_guess = (amplitude, lifetime, background)
        try:
            params, _covariance = curve_fit(exponential_model, _times, _counts, p0=initial_guess)
            return (params, _covariance)
        except:
            print('Fitting failed. Try adjusting initial parameters.')
            return (initial_guess, None)

    def plot_lifetime_decay(times, counts, fitted_params=None, log_scale=True, residuals=True, title='Fluorescence Lifetime Decay'):
        """
        Plot the lifetime decay curve and optionally the fit.
    
        Parameters:
        -----------
        times : numpy array
            Time points
        counts : numpy array
            Count data
        fitted_params : tuple or None
            Fitted parameters (amplitude, lifetime, background)
        log_scale : bool
            Whether to use log scale for y-axis
        residuals : bool
            Whether to plot residuals (only if fitted_params is provided)
        title : str
            Plot title
        """
        if residuals and _fitted_params is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(_times, _counts, 'b.', alpha=0.5, label='Data')
        if _fitted_params is not None:
            amplitude, lifetime, background = _fitted_params
            fitted_curve = exponential_model(_times, amplitude, lifetime, background)
            ax1.plot(_times, fitted_curve, 'r-', label=f'Fit: τ = {lifetime:.3f} ns')
            text_info = f'Fitted τ = {lifetime:.3f} ns\nAmplitude = {amplitude:.1f}\nBackground = {background:.1f}'
            ax1.text(0.98, 0.98, text_info, transform=ax1.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Counts')
        ax1.set_title(title)
        ax1.legend()
        if log_scale:
            ax1.set_yscale('log')
        if residuals and _fitted_params is not None:
            fitted_curve = exponential_model(_times, *_fitted_params)
            res = _counts - fitted_curve
            ax2.plot(_times, res, 'g.', alpha=0.5)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('Residuals')
        plt.tight_layout()
        plt.show()
        return fig

    def simulate_multi_exponential_decay(num_photons, lifetimes, amplitudes, time_range=(0, 10), time_bins=1000, background=0):
        """
        Simulate a multi-exponential fluorescence decay curve.
    
        Parameters:
        -----------
        num_photons : int
            Total number of photons to simulate
        lifetimes : list of float
            Lifetimes in nanoseconds for each component
        amplitudes : list of float
            Relative amplitudes for each component (will be normalized)
        time_range : tuple
            (min_time, max_time) in nanoseconds
        time_bins : int
            Number of time bins to use
        background : float
            Expected background counts per bin
        
        Returns:
        --------
        times : numpy array
            Time points of the simulation
        counts : numpy array
            Photon counts at each time point
        """
        min_time, max_time = _time_range
        norm_amplitudes = np.array(amplitudes) / np.sum(amplitudes)
        component_photons = np.random.multinomial(_num_photons, norm_amplitudes)
        all_photon_times = np.array([])
        for i, (lifetime, n_photons) in enumerate(zip(lifetimes, component_photons)):
            if n_photons > 0:
                photon_times = np.random.exponential(scale=lifetime, size=n_photons)
                all_photon_times = np.append(all_photon_times, photon_times)
            print(f'len all photons:{len(all_photon_times)}')
        all_photon_times = all_photon_times[all_photon_times <= max_time]
        bin_edges = np.linspace(min_time, max_time, _time_bins + 1)
        _counts, _ = np.histogram(all_photon_times, bins=bin_edges)
        _times = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if background > 0:
            bg_counts = np.random.poisson(background, size=_time_bins)
            _counts = _counts + bg_counts
        return (_times, _counts)
    if __name__ == '__main__':
        _num_photons = 10000
        _true_lifetime = 1.2
        _time_range = (0, 12.5)
        _time_bins = 256
        _background_level = 0
        _times, _counts = generate_lifetime_decay_photon_simulation(num_photons=_num_photons, lifetime=_true_lifetime, time_range=_time_range, time_bins=_time_bins, background=_background_level)
        _fitted_params, _covariance = fit_lifetime_decay(_times, _counts)
        if _covariance is not None:
            _perr = np.sqrt(np.diag(_covariance))
            _fitted_lifetime = _fitted_params[1]
            _lifetime_error = _perr[1]
            _cv_percent = _lifetime_error / _fitted_lifetime * 100
            print(f'True lifetime: {_true_lifetime} ns')
            print(f'Fitted lifetime: {_fitted_params[1]:.3f} ± {_perr[1]:.3f} ns')
            print(f'Total photons detected: {np.sum(_counts)}')
            print(f'Coefficient of variation: {_cv_percent:.2f}%')
        else:
            print(f'True lifetime: {_true_lifetime} ns')
            print(f'Fitting failed to converge')
        '\n    # Plot the results\n    plot_lifetime_decay(times, counts, fitted_params, log_scale=True, residuals=True,\n                       title=f"Fluorescence Decay (τ = {true_lifetime} ns, {num_photons} photons)")\n    \n    # Example of multi-exponential decay simulation\n    print("\nSimulating multi-exponential decay...")\n    times_multi, counts_multi = simulate_multi_exponential_decay(\n        num_photons=num_photons,\n        lifetimes=[1.2, 3.5],  # Two lifetime components\n        amplitudes=[0.7, 0.3],  # 70% first component, 30% second component\n        time_range=time_range,\n        time_bins=time_bins,\n        background=background_level\n    )\n    \n    plt.figure(figsize=(10, 6))\n    plt.plot(times_multi, counts_multi, \'b.\', alpha=0.5)\n    plt.yscale(\'log\')\n    plt.xlabel(\'Time (ns)\')\n    plt.ylabel(\'Counts\')\n    plt.title(\'Multi-exponential Decay (τ₁=1.2ns, τ₂=3.5ns)\')\n    plt.tight_layout()\n    plt.show()\n    '
        export_decay_data_as_3d_tiff(_times, _counts, f'E:\\Projects\\Fluorescein_Quenching\\python_simulations\\SINGLE_t1_{_true_lifetime}_my_decay_data_4.tiff', box_size=(40, 40))
    return (
        curve_fit,
        fit_lifetime_decay,
        generate_lifetime_decay_photon_simulation,
        np,
        plt,
        simulate_multi_exponential_decay,
    )


@app.cell
def _(np):
    _bincalc = np.rint(np.linspace(2, 256, 15)).astype(int)
    print(_bincalc)
    return


@app.cell
def _(fit_lifetime_decay, generate_lifetime_decay_photon_simulation, np, plt):
    datastore = []
    _bincalc = np.rint(2 + (256 - 2) * np.linspace(0, 1, 15) ** 2).astype(int)
    trials = np.linspace(1, 5, 100)
    taurepeats = np.linspace(1, 5, 10)
    for c in trials:
        for t in taurepeats:
            all_data = []
            for b in _bincalc:
                if __name__ == '__main__':
                    _num_photons = 10000
                    _true_lifetime = t
                    _time_range = (0, 12.5)
                    _time_bins = b
                    _background_level = 0
                    _times, _counts = generate_lifetime_decay_photon_simulation(num_photons=_num_photons, lifetime=_true_lifetime, time_range=_time_range, time_bins=_time_bins, background=_background_level)
                    _fitted_params, _covariance = fit_lifetime_decay(_times, _counts)
                    if _covariance is not None:
                        _perr = np.sqrt(np.diag(_covariance))
                        _fitted_lifetime = _fitted_params[1]
                        _lifetime_error = _perr[1]
                        _cv_percent = _lifetime_error / _fitted_lifetime * 100
                        print(f'True lifetime: {_true_lifetime} ns')
                        print(f'Fitted lifetime: {_fitted_params[1]:.3f} ± {_perr[1]:.3f} ns')
                        print(f'Total photons detected: {np.sum(_counts)} why not 10000???')
                        print(f'Coefficient of variation: {_cv_percent:.2f}%')
                        datastore.append({'true lifetime': _true_lifetime, 'fitted lifetime': _fitted_lifetime, 'time_bins': _time_bins, 'cv_percent': _cv_percent})
                    else:
                        print(f'True lifetime: {_true_lifetime} ns')
                        print(f'Fitting failed to converge')
                    all_data.append({'times': _times, 'counts': _counts, 'fitted_params': _fitted_params, 'bin_size': _time_bins})
            '    \n        plt.figure(figsize=(12, 8))\n\n        # Create a color cycle for different curves\n        colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))\n\n        for i, data in enumerate(all_data):\n            times = data[\'times\']\n            counts = data[\'counts\']\n            fitted_params = data[\'fitted_params\']\n            bin_size = data[\'bin_size\']\n            \n            # Plot data points\n            plt.semilogy(times, counts, \'o\', label=f\'Bin size: {bin_size}\', \n                            color=colors[i], markersize=4, alpha=0.7)\n            \n            # Plot fitted curve if fit was successful\n            if fitted_params is not None:\n                fitted_curve = exponential_model(times, *fitted_params)\n                plt.semilogy(times, fitted_curve, \'-\', linewidth=2, \n                                color=colors[i], alpha=0.9)\n\n        plt.xlabel(\'Time (ns)\')\n        plt.ylabel(\'Counts (log scale)\')\n        plt.title(f"Fluorescence Decay (τ = {true_lifetime} ns, {num_photons} photons)")\n        plt.grid(True, alpha=0.3)\n        plt.legend()\n        plt.tight_layout()\n        plt.show()\n        '
    fitted_lifetimes = [item['fitted lifetime'] for item in datastore]
    cv_percents = [item['cv_percent'] for item in datastore]
    bin_sizes = [item['time_bins'] for item in datastore]
    from collections import defaultdict
    data_by_true_lifetime = defaultdict(list)
    for item in datastore:
        _true_lifetime = item['true lifetime']
        fitted = item['fitted lifetime']
        bin = item['time_bins']
        data_by_true_lifetime[_true_lifetime, bin].append(fitted)
    true = []
    cvbytrue = []
    binbytrue = []
    numcvcount = 0
    for key, fitted in data_by_true_lifetime.items():
        _true_lifetime, bin = key
        mean_fittedbytrue = np.mean(fitted)
        std_fittedbytrue = np.std(fitted)
        cvbytruecalc = std_fittedbytrue / mean_fittedbytrue
        true.append(_true_lifetime)
        cvbytrue.append(cvbytruecalc)
        binbytrue.append(bin)
        numcvcount += 1
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(fitted_lifetimes, cv_percents, c=bin_sizes, s=50, alpha=0.7, cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Bins')
    plt.xlabel('Fitted Lifetime (ns)')
    plt.ylabel('Coefficient of Variation (%)')
    plt.title('CV per curve? fot fitting? vs. Fitted Lifetime')
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.scatter(true, cvbytrue, c=binbytrue)
    cbar.set_label('time bins')
    plt.colorbar()
    plt.xlabel('fit lifetime (ns)')
    plt.ylabel('cv of trials')
    plt.title('cv of fitted lifetime trials')
    plt.show
    print(numcvcount)
    return


@app.cell
def _(curve_fit, np, plt, simulate_multi_exponential_decay):
    def multi_exponential_model(t, *params):
        """
        Multi-exponential decay model with background.
    
        Parameters:
        t : time
        params : tuple containing:
            - amplitudes (a1, a2, ..., an)
            - lifetimes (tau1, tau2, ..., taun)
            - background constant c
    
        Returns:
        --------
        Decay curve values at times t
        """
        n_components = (len(params) - 1) // 2
        amplitudes = params[:n_components]
        lifetimes = params[n_components:2 * n_components]
        background = params[-1]
        result = background
        for i in range(n_components):
            result += amplitudes[i] * np.exp(-t / lifetimes[i])
        print('USING MULTI EXP MODEL')
        return result

    def fit_multi_exponential_decay(times, counts, n_components, initial_guess=None):
        """
        Fit the lifetime decay curve with a multi-exponential model.
    
        Parameters:
        -----------
        times : numpy array
            Time points
        counts : numpy array
            Count data
        n_components : int
            Number of exponential components to fit
        initial_guess : tuple or None
            Initial parameters (a1, a2, ..., an, tau1, tau2, ..., taun, background)
        
        Returns:
        --------
        params : tuple
            Fitted parameters (a1, a2, ..., an, tau1, tau2, ..., taun, background)
        covariance : array
            Covariance matrix of the fit
        """
        if initial_guess is None:
            background = np.min(_counts)
            amplitude_total = np.max(_counts) - background
            amplitudes = [amplitude_total / n_components] * n_components
            min_lifetime = _times[1] - _times[0]
            max_lifetime = (_times[-1] - _times[0]) / 3
            lifetimes = np.logspace(np.log10(min_lifetime), np.log10(max_lifetime), n_components)
            initial_guess = tuple(amplitudes) + tuple(lifetimes) + (background,)
        try:
            params, _covariance = curve_fit(multi_exponential_model, _times, _counts, p0=initial_guess)
            return (params, _covariance)
        except Exception as e:
            print(f'Fitting failed. Error: {e}')
            print('Try adjusting initial parameters.')
            return (initial_guess, None)

    def analyze_multi_exponential_decay(times, counts, n_components, initial_guess=None):
        """
        Analyze multi-exponential decay data and return fitted parameters and derived values.
    
        Parameters:
        -----------
        times : numpy array
            Time points
        counts : numpy array
            Count data
        n_components : int
            Number of exponential components to fit
        initial_guess : tuple or None
            Initial parameters (a1, a2, ..., an, tau1, tau2, ..., taun, background)
        
        Returns:
        --------
        result_dict : dict
            Dictionary containing fitted parameters and derived values:
            - 'fitted_params': tuple of all fitted parameters
            - 'amplitudes': numpy array of fitted amplitudes
            - 'lifetimes': numpy array of fitted lifetimes
            - 'background': fitted background value
            - 'fractional_amplitudes': relative contribution of each component
            - 'average_lifetime': amplitude-weighted average lifetime
            - 'errors': standard errors of parameters (if fit was successful)
            - 'fitted_curve': array of fitted values at the same time points
        """
        _fitted_params, _covariance = fit_multi_exponential_decay(_times, _counts, n_components, initial_guess)
        amplitudes = _fitted_params[:n_components]
        lifetimes = _fitted_params[n_components:2 * n_components]
        background = _fitted_params[-1]
        fitted_curve = multi_exponential_model(_times, *_fitted_params)
        print(fitted_curve)
        total_amplitude = np.sum(amplitudes)
        fractional_amplitudes = amplitudes / total_amplitude
        average_lifetime = np.sum(fractional_amplitudes * lifetimes)
        intensity_values = amplitudes * lifetimes
        intensity_fractions = intensity_values / np.sum(intensity_values)
        result = {'fitted_params': _fitted_params, 'amplitudes': amplitudes, 'lifetimes': lifetimes, 'background': background, 'fractional_amplitudes': fractional_amplitudes, 'intensity_fractions': intensity_fractions, 'average_lifetime': average_lifetime, 'fitted_curve': fitted_curve}
        if _covariance is not None:
            _perr = np.sqrt(np.diag(_covariance))
            result['errors'] = _perr
            result['amplitude_errors'] = _perr[:n_components]
            result['lifetime_errors'] = _perr[n_components:2 * n_components]
            result['background_error'] = _perr[-1]
            residuals = _counts - fitted_curve
            chi_squared = np.sum(residuals ** 2 / (_counts + 1))
            reduced_chi_squared = chi_squared / (len(_counts) - len(_fitted_params))
            result['chi_squared'] = chi_squared
            result['reduced_chi_squared'] = reduced_chi_squared
        return result

    def plot_multi_exponential_decay(times, counts, analysis_result, log_scale=True, residuals=True, title='Multi-exponential Fluorescence Decay'):
        """
        Plot the multi-exponential lifetime decay curve with fit and components.
    
        Parameters:
        -----------
        times : numpy array
            Time points
        counts : numpy array
            Count data
        analysis_result : dict
            The result dictionary from analyze_multi_exponential_decay
        log_scale : bool
            Whether to use log scale for y-axis
        residuals : bool
            Whether to plot residuals
        title : str
            Plot title
    
        Returns:
        --------
        fig : matplotlib figure
            The figure object containing the plot
        """
        fitted_curve = analysis_result['fitted_curve']
        amplitudes = analysis_result['amplitudes']
        lifetimes = analysis_result['lifetimes']
        background = analysis_result['background']
        n_components = len(lifetimes)
        if residuals:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(_times, _counts, 'b.', alpha=0.5, label='Data')
        ax1.plot(_times, fitted_curve, 'r-', linewidth=2, label='Overall Fit')
        colors = ['g', 'm', 'c', 'y', 'orange']
        for i in range(n_components):
            component = amplitudes[i] * np.exp(-_times / lifetimes[i]) + background
            ax1.plot(_times, component, '--', color=colors[i % len(colors)], alpha=0.7, label=f'Comp {i + 1}: τ = {lifetimes[i]:.3f} ns')
        ax1.axhline(y=background, color='k', linestyle=':', alpha=0.5, label=f'Background = {background:.1f}')
        text_lines = [f"Component {i + 1}: τ = {lifetimes[i]:.3f} ns, A = {amplitudes[i]:.1f}, Frac = {analysis_result['fractional_amplitudes'][i] * 100:.1f}%" for i in range(n_components)]
        if 'errors' in analysis_result:
            lifetime_errors = analysis_result['lifetime_errors']
            for i in range(n_components):
                text_lines[i] += f' ± {lifetime_errors[i]:.3f} ns'
        text_lines.append(f"Average τ = {analysis_result['average_lifetime']:.3f} ns")
        if 'reduced_chi_squared' in analysis_result:
            text_lines.append(f"Reduced χ² = {analysis_result['reduced_chi_squared']:.3f}")
        ax1.text(0.98, 0.98, '\n'.join(text_lines), transform=ax1.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Counts')
        ax1.set_title(title)
        ax1.legend(loc='upper right')
        if log_scale:
            ax1.set_yscale('log')
        if residuals:
            res = _counts - fitted_curve
            ax2.plot(_times, res, 'g.', alpha=0.5)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('Residuals')
            if 'errors' in analysis_result:
                res_mean = np.mean(res)
                res_std = np.std(res)
                ax2.text(0.98, 0.98, f'Mean = {res_mean:.1f}\nStd = {res_std:.1f}', transform=ax2.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        return fig
    if __name__ == '__main__':
        print('\nAnalyzing multi-exponential decay...')
        _num_photons = 10000
        true_lifetimes = [0.4, 1.2]
        true_amplitudes = [0.7, 0.3]
        _time_range = (0, 12.5)
        _time_bins = 256
        _background_level = 0
        times_multi, counts_multi = simulate_multi_exponential_decay(num_photons=_num_photons, lifetimes=true_lifetimes, amplitudes=true_amplitudes, time_range=_time_range, time_bins=_time_bins, background=_background_level)
        initial_guess = (_num_photons * 0.3 * 0.1, _num_photons * 0.7 * 0.1, 0.3, 1.1, _background_level)
        analysis_result = analyze_multi_exponential_decay(times_multi, counts_multi, n_components=2, initial_guess=initial_guess)
        print('\nBiexponential Decay Analysis Results:')
        print(f'True lifetimes: {true_lifetimes} ns')
        print(f"Fitted lifetimes: {analysis_result['lifetimes']} ns")
        if 'lifetime_errors' in analysis_result:
            print(f"Lifetime errors: {analysis_result['lifetime_errors']} ns")
        print(f'True amplitude fractions: {true_amplitudes}')
        print(f"Fitted amplitude fractions: {analysis_result['fractional_amplitudes']}")
        print(f"Intensity fractions: {analysis_result['intensity_fractions']}")
        print(f"Average lifetime: {analysis_result['average_lifetime']:.3f} ns")
        print(f'Total photons detected: {np.sum(counts_multi)} multi')
        if 'reduced_chi_squared' in analysis_result:
            print(f"Reduced chi-squared: {analysis_result['reduced_chi_squared']:.3f}")
        fig = plot_multi_exponential_decay(times_multi, counts_multi, analysis_result, log_scale=True, residuals=True, title=f'Biexponential Decay Analysis (τ₁={true_lifetimes[0]} ns, τ₂={true_lifetimes[1]} ns)')
        '\n    # Try fitting with different number of components\n    print("\nTrying to fit with 3 components (when true model is 2 components)...")\n    analysis_result_3comp = analyze_multi_exponential_decay(\n        times_multi, \n        counts_multi, \n        n_components=3\n    )\n    \n    # Plot the 3-component fit for comparison\n    fig3 = plot_multi_exponential_decay(\n        times_multi, \n        counts_multi, \n        analysis_result_3comp,\n        log_scale=True, \n        residuals=True,\n        title="Testing 3-Component Fit (when true model is 2 components)"\n    )\n    '
        plt.show()
    return counts_multi, times_multi, true_lifetimes


@app.cell
def _(counts_multi, np, times_multi, true_lifetimes):
    import json

    class NumpyEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.number):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    def export_decay_data_as_3d_tiff(times, counts, filename='decay_data_3d.tiff', normalize=False, time_dimension_first=True, box_size=(40, 40)):
        """
        Export fluorescence lifetime decay data as a 3D TIFF image stack with the same decay 
        curve repeated across a 2D image of specified size.
   
        Parameters:
        -----------
        times : numpy array
            Time points of the decay curve (nanoseconds)
        counts : numpy array
            Count data of the decay curve
        filename : str
            Output filename for the 3D TIFF image
        normalize : bool
            Whether to normalize the counts to 0-65535 range (16-bit)
        time_dimension_first : bool
            If True, time is the first dimension (Z), otherwise it's the last
        box_size : tuple
            Size of the 2D image (width, height) to repeat the decay curve across
   
        Returns:
        --------
        str : Path to the saved file
        """
        try:
            import tifffile
        except ImportError:
            print("This function requires the 'tifffile' package. Please install it with:")
            print('pip install tifffile')
            return None
        _times = np.asarray(_times)
        _counts = np.asarray(_counts)
        width, height = box_size
        if time_dimension_first:
            single_slice = np.ones((width, height))
            data_3d = np.zeros((len(_counts), width, height), dtype=float)
            for t in range(len(_counts)):
                data_3d[t, :, :] = single_slice * _counts[t]
        else:
            data_3d = np.zeros((width, height, len(_counts)), dtype=float)
            for x in range(width):
                for y in range(height):
                    data_3d[x, y, :] = _counts
        if normalize:
            min_val = np.min(data_3d)
            max_val = np.max(data_3d)
            if max_val > min_val:
                data_3d = ((data_3d - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
            else:
                data_3d = np.zeros_like(data_3d, dtype=np.uint16)
        blankarr = np.zeros((256, 256, 256))
        print(data_3d.shape)
        blankarr[:, 127 - box_size[0] // 2:127 + box_size[0] // 2, 127 - box_size[1] // 2:127 + box_size[1] // 2] = data_3d
        data_3d = blankarr
        metadata = {'time_points': _times, 'time_unit': 'nanoseconds', 'min_time': _times[0], 'max_time': _times[-1], 'time_step': _times[1] - _times[0] if len(_times) > 1 else 0, 'box_size': box_size, 'description': 'Decay curve repeated across a 2D image'}
        metadata_str = json.dumps(metadata, cls=NumpyEncoder)
        tifffile.imwrite(filename, data_3d, metadata={'time_info': metadata_str}, imagej=False, resolution=(1.0, 1.0), photometric='minisblack', compression='lzw')
        print(f'Decay data exported as 3D TIFF with dimensions {data_3d.shape}: {filename}')
        csv_filename = filename.replace('.tiff', '.csv').replace('.tif', '.csv')
        np.savetxt(csv_filename, np.column_stack((_times, _counts)), delimiter=',', header='Time(ns),Counts', comments='')
        print(f'Raw decay data saved as CSV: {csv_filename}')
        return filename
    export_decay_data_as_3d_tiff(times_multi, counts_multi, f'E:\\Projects\\Fluorescein_Quenching\\python_simulations\\MULTI_t1_{true_lifetimes[0]}_t2_{true_lifetimes[1]}_my_decay_data.tiff', box_size=(40, 40))
    return (export_decay_data_as_3d_tiff,)


if __name__ == "__main__":
    app.run()
