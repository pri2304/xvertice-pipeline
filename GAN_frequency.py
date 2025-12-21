import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import time
from PIL import Image
from scipy.stats import linregress


class GANMonitor:
    @staticmethod
    def calculate_azimuthal_average(magnitude_spectrum):
        """
        Calculates the average pixel intensity for each radius (ring) from the center.
        This converts the 2D spectrum into a 1D curve.
        """
        h, w = magnitude_spectrum.shape
        cx, cy = w // 2, h // 2

        # Create a grid of coordinates relative to center
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        r = np.sqrt(x ** 2 + y ** 2).astype(int)

        # Bin the pixels by radius
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())

        # Avoid division by zero
        radial_profile = tbin / np.maximum(nr, 1)

        return radial_profile

    @staticmethod
    def analyze(image_path):
        try:
            # Load Image (Grayscale)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                pil_img = Image.open(image_path).convert('L')
                img = np.array(pil_img)

            # FFT Transform (Spatial -> Frequency)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)

            # We use absolute value because FFT returns complex numbers
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

            # This simplifies the analysis from a "2D Starfield" to a "1D Curve"
            profile = GANMonitor.calculate_azimuthal_average(magnitude_spectrum)

            profile_norm = (profile - np.min(profile)) / (np.max(profile) - np.min(profile))

            cutoff = int(len(profile_norm) * 0.7)
            high_freq_tail = profile_norm[cutoff:]

            tail_energy = np.mean(high_freq_tail)

            fluctuation = np.std(np.diff(high_freq_tail))

            x_axis = np.log(np.arange(1, len(profile)) + 1)
            y_axis = np.log(profile[1:] + 1e-8)

            # Simple regression
            slope, intercept, r_value, p_value, std_err = linregress(x_axis, y_axis)

            metrics = {
                "high_freq_energy": round(float(tail_energy), 4),
                "tail_roughness": round(float(fluctuation), 5),
                "power_law_fit_error": round(float(std_err), 5)
            }

            # 6. Generate Graph
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot 1: 2D Spectrum (Visual Check)
            ax1.imshow(magnitude_spectrum, cmap='inferno')
            ax1.set_title('2D Frequency Spectrum')
            ax1.axis('off')

            # Plot 2: 1D Profile (Analytical Check)
            ax2.plot(profile_norm, color='cyan', label='Image Profile')

            # Highlight the High-Freq Tail
            ax2.axvline(x=cutoff, color='red', linestyle='--', label='High Freq Start')
            ax2.set_title(f'1D Power Profile (Roughness: {metrics["tail_roughness"]})')
            ax2.set_xlabel('Frequency (Radius)')
            ax2.set_ylabel('Power (Normalized)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            return metrics, buf

        except Exception as e:
            print(f"Error performing GAN Analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, None


if __name__ == "__main__":
    input_path = "Testing Images/testcase20.jpeg"
    start_time = time.time()

    gan_metrics, gan_graph = GANMonitor.analyze(input_path)

    duration = time.time() - start_time
    print(f"GAN Test took {duration:.4f} seconds")
    print(gan_metrics)

    if gan_graph:
        with open("Results/gan_result.png", "wb") as f:
            f.write(gan_graph.getbuffer())