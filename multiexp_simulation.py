import numpy as np
import matplotlib.pyplot as plt


def simulate_multi_exponential_decay(
    num_photons, lifetimes, fractions, time_range, time_bins=256, background=0
):
    min_time, max_time = time_range
    norm_amplitudes = np.array(fractions) / np.sum(fractions)
    component_photons = np.random.multinomial(num_photons, norm_amplitudes)
    all_photon_times = np.array([])
    for i, (lifetime, n_photons) in enumerate( zip(lifetimes,
                                                   component_photons)):
        if n_photons > 0:
            photon_times = np.random.exponential(scale=lifetime,
                                                 size=n_photons)
            all_photon_times = np.append(all_photon_times, photon_times)
    all_photon_times = all_photon_times[all_photon_times <= max_time]
    bin_edges = np.linspace(min_time, max_time, time_bins + 1)
    counts, _ = np.histogram(all_photon_times, bins=bin_edges)
    times = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if background > 0:
        bg_counts = np.random.poisson(background, size=time_bins)
        counts = counts + bg_counts
    return times, counts


def plot_multi_exponential_decay(
    times,
    counts,
    lifetimes,
    background,
    amplitudes,
    log_scale=True,
    title="Multi-exponential Fluorescence Decay",
    ):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(times, counts, "b.", alpha=0.5, label="Data")
    n_components = len(lifetimes)
    colors = ["g", "m", "c", "y", "orange"]  # Colors for different components
    for i in range(n_components):
        component = (
            max(counts) * amplitudes[i] * np.exp(-times / lifetimes[i]) + background
        )
        ax1.plot(
            times,
            component,
            "--",
            color=colors[i % len(colors)],
            alpha=0.7,
            label=f"Comp {i + 1}: τ = {lifetimes[i]:.3f} ns, a = {amplitudes[i]}",
        )
    ax1.axhline(
        y=background,
        color="k",
        linestyle=":",
        alpha=0.5,
        label=f"Background = {background:.1f}",
    )
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Counts")
    ax1.set_title(title)
    ax1.legend(loc="upper right")

    if log_scale:
        ax1.set_yscale("log")

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create a new biexponential decay example with known parameters
    num_photons = 5
    true_lifetimes = [1.1, 0]  # Two lifetime components (in ns)
    true_amplitudes = [1, 0]  # Relative amplitudes (70% and 30%)
    time_range = (0, 12.5)  # nanoseconds
    time_bins = 256
    # background_level = (num_photons*0.2)/256 #background counts per bin
    background_level = 0

    # Simulate the biexponential decay
    times_multi, counts_multi = simulate_multi_exponential_decay(
        num_photons=num_photons,
        lifetimes=true_lifetimes,
        fractions=true_amplitudes,
        time_range=time_range,
        time_bins=time_bins,
        background=background_level,
    )

    # Display the results
    # print("\nBiexponential Decay Analysis Results:")
    print(f"True lifetimes: {true_lifetimes} ns")
    print(f"True amplitude fractions: {true_amplitudes}")
    print(f"Total photons detected: {np.sum(counts_multi)} multi")

    # Plot the results
    fig = plot_multi_exponential_decay(
        times_multi,
        counts_multi,
        lifetimes=true_lifetimes,
        amplitudes=true_amplitudes,
        background=background_level,
        log_scale=False,
        title=f"Biexponential Decay Analysis (τ₁={true_lifetimes[0]} ns, τ₂={true_lifetimes[1]} ns)",
    )

    plt.show()
