import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FourierVisualizer:
    def __init__(self, n_points=200):
        np.random.seed(42)
        self.time = np.arange(n_points)
        # Simulated signal: mix of two frequencies + noise
        self.signal = (np.sin(2 * np.pi * self.time / 20) +
                       0.5 * np.sin(2 * np.pi * self.time / 5) +
                       np.random.normal(0, 0.3, n_points))
        self.fft = np.fft.fft(self.signal)
        self.freq = np.fft.fftfreq(n_points)

    def animate(self, filename="fourier_transform.gif"):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        plt.tight_layout(pad=3)

        def update(frame):
            for ax in axs:
                ax.clear()

            # Top plot: time series
            axs[0].plot(self.time, self.signal, color="blue", alpha=0.6, label="Original Signal")
            # Reconstructed signal with first N frequencies
            partial_fft = np.zeros_like(self.fft)
            partial_fft[:frame] = self.fft[:frame]
            partial_fft[-frame:] = self.fft[-frame:]
            reconstructed = np.fft.ifft(partial_fft).real
            axs[0].plot(self.time, reconstructed, color="red", label=f"Reconstructed ({frame} freq.)")
            axs[0].set_title("Time Series and Reconstructed Signal")
            axs[0].legend()
            axs[0].grid(True, linestyle="--", alpha=0.6)

            # Bottom plot: frequency spectrum
            axs[1].stem(self.freq[:len(self.freq)//2], np.abs(self.fft)[:len(self.freq)//2],
                        linefmt="grey", markerfmt=" ", basefmt=" ")
            axs[1].set_xlim(0, 0.5)
            axs[1].set_title("Frequency Spectrum (Fourier Transform)")
            axs[1].set_xlabel("Frequency")
            axs[1].set_ylabel("Magnitude")

        ani = animation.FuncAnimation(fig, update, frames=30, interval=300, repeat=False)
        ani.save(filename, writer="pillow")
        plt.close(fig)


if __name__ == "__main__":
    viz = FourierVisualizer()
    viz.animate("fourier_transform.gif")
