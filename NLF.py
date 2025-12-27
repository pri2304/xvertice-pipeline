import numpy as np
from PIL import Image, ImageFilter, ImageChops
import matplotlib.pyplot as plt
import io
import time
from scipy.optimize import curve_fit
from skimage.util import view_as_blocks


class NLF:
    @staticmethod
    def nlf_analyze(image_path, block_size=32):
        try:
            # Load and Convert to Grayscale
            raw_image = Image.open(image_path).convert('L')
            img_arr = np.array(raw_image, dtype=np.float32)

            h, w = img_arr.shape
            new_h, new_w = (h // block_size) * block_size, (w // block_size) * block_size

            # If image is too small, return early
            if new_h < block_size or new_w < block_size:
                return {"error": "Image too small"}, None

            img_arr = img_arr[:new_h, :new_w]

            # Subtracting median filter gives us the "texture/noise" layer
            median_blurred = raw_image.filter(ImageFilter.MedianFilter(size=3))
            median_arr = np.array(median_blurred, dtype=np.float32)[:new_h, :new_w]
            noise_arr = img_arr - median_arr

            img_blocks = view_as_blocks(img_arr, block_shape=(block_size, block_size))
            noise_blocks = view_as_blocks(noise_arr, block_shape=(block_size, block_size))

            # Calculate mean and variance for ALL blocks at once
            means = np.mean(img_blocks, axis=(2, 3)).flatten()
            variances = np.var(noise_blocks, axis=(2, 3)).flatten()

            var_threshold = np.percentile(variances, 95)
            mask = (means > 5) & (means < 250) & (variances < var_threshold)

            clean_means = means[mask]
            clean_variances = variances[mask]

            def linear_model(x, a, b):
                return a * x + b

            popt, _ = curve_fit(linear_model, clean_means, clean_variances)
            slope, intercept = popt

            expected_variances = linear_model(clean_means, slope, intercept)
            rmse = np.sqrt(np.mean((clean_variances - expected_variances) ** 2))

            global_outlier_mask = np.abs(clean_variances - expected_variances) > (4 * rmse)
            outlier_ratio = (np.sum(global_outlier_mask) / len(clean_means)) * 100

            metrics = {
                "noise_fit_rmse": float(rmse),
                "slope": float(slope),
                "intercept": float(intercept),
                "outlier_ratio": round(float(outlier_ratio), 2),
            }

            plt.figure(figsize=(8, 5))

            # Plot all points (grey/faint)
            plt.scatter(means, variances, alpha=0.1, s=5, color='gray', label='All Blocks')

            # Plot clean points (Blue)
            plt.scatter(clean_means[~global_outlier_mask], clean_variances[~global_outlier_mask],
                        alpha=0.6, s=10, color='tab:blue', label='Consistent Blocks')

            # Plot outliers (Red)
            if np.sum(global_outlier_mask) > 0:
                plt.scatter(clean_means[global_outlier_mask], clean_variances[global_outlier_mask],
                            alpha=0.8, s=15, color='tab:red', label='Outliers')

            # Plot Line
            x_range = np.linspace(0, 255, 100)
            plt.plot(x_range, linear_model(x_range, slope, intercept),
                     color='green', linewidth=2, label='Noise Profile')

            plt.title(f'(Outliers: {outlier_ratio}%)')
            plt.xlabel('Brightness')
            plt.ylabel('Noise Variance')
            plt.legend()
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            return metrics, buf

        except Exception as e:
            print(f"Error performing NLF Analysis: {e}")
            return None, None


if __name__ == "__main__":
    input_path = "/home/pri/Dataset/Fake/Faceforensics/DeepFakeDetection/07_20__outside_talking_pan_laughing__KV6Q7D6C_frame_0.jpg"
    start_time = time.time()

    nlf_metrics, nlf_graph_buffer = NLF.nlf_analyze(input_path)

    end_time = time.time()
    duration = end_time - start_time
    print(f"NLF Test took {duration:.4f} seconds")
    print(nlf_metrics)

    if nlf_graph_buffer:
        with open("Results/nlf_result_graph.png", "wb") as f:
            f.write(nlf_graph_buffer.getbuffer())