

def multi_exponential_model(t, lifetimes,amplitudes,background):
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
    n_components = (len(lifetimes))
    result = background
    for i in range(n_components):
        result += amplitudes[i] * np.exp(-t / lifetimes[i])
    print('USING MULTI EXP MODEL')
    return result


if __name__ == "__main__":
    
    #direct estimate (i.e. not randomized)
    lifetimecurve = multi_exponential_model(times_multi,true_lifetimes, true_amplitudes, background_level)
    
    fig = plot_multi_exponential_decay(
            times_multi, 
            lifetimecurve, 
            lifetimes=true_lifetimes,
            amplitudes=true_amplitudes,
            background=background_level,
            log_scale=False, 
            title=f"Not randomized biexponential (τ₁={true_lifetimes[0]} ns, τ₂={true_lifetimes[1]} ns)"
        )