import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import time
from PIL import Image


class DCT:
    @staticmethod
    def dct_analyze(image_path):
        try:
            # Load Image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                pil_img = Image.open(image_path).convert('L')
                img = np.array(pil_img)

            h, w = img.shape

            # Crop to be multiple of 8
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8
            img = img[:new_h, :new_w]
            img_float = np.float32(img)

            # Perform Block-wise DCT
            blocks = [img_float[j:j + 8, i:i + 8] for j in range(0, new_h, 8) for i in range(0, new_w, 8)]

            dct_coefficients = []

            # Variables for Energy Ratio
            total_energy = 0.0
            hf_energy = 0.0

            for block in blocks:
                # Apply DCT
                dct_block = cv2.dct(block)

                block_abs = np.abs(dct_block)
                total_energy += np.sum(block_abs)
                hf_energy += np.sum(block_abs[4:, 4:])  # Bottom-right 4x4 sub-block

                # Flatten to 1D array
                flat = dct_block.flatten()

                # Skip DC Coefficient (index 0)
                dct_coefficients.extend(flat[1:])

            dct_coefficients = np.array(dct_coefficients)

            # Metric: Periodicity (Your existing logic)
            hist, bins = np.histogram(dct_coefficients, bins=100, range=(-50, 50))
            diffs = np.abs(np.diff(hist))
            periodicity_score = np.sum(diffs) / np.sum(hist)

            # Metric: Sparsity (Percentage of Zero Coefficients)
            # JPEG compression creates many zeros. High quality = Low sparsity.
            # Fake/Upscaled images might have weird sparsity.
            zero_count = np.sum(dct_coefficients == 0)
            sparsity_score = zero_count / len(dct_coefficients)

            # Metric: High Frequency Energy Ratio
            # Deepfakes/AI often struggle to put energy in the high frequencies (texture).
            hf_ratio = hf_energy / total_energy if total_energy > 0 else 0

            # Metric: Benford's Law Divergence
            # Natural images follow Benford's law (1 appears most, 9 appears least).
            # Check the first digit of every coefficient.

            benford_coeffs = np.abs(dct_coefficients)
            benford_coeffs = benford_coeffs[benford_coeffs >= 1]

            if len(benford_coeffs) > 0:
                first_digits = np.floor(benford_coeffs / (10 ** np.floor(np.log10(benford_coeffs)))).astype(int)

                # Count occurrences of 1 through 9
                digit_counts = np.zeros(9)
                for d in range(1, 10):
                    digit_counts[d - 1] = np.sum(first_digits == d)

                # Calculate Observed Probabilities
                observed_probs = digit_counts / np.sum(digit_counts)

                # Calculate Expected Probabilities (Benford's Law: log10(1 + 1/d))
                expected_probs = np.log10(1 + 1 / np.arange(1, 10))

                # Divergence Score (Mean Squared Error between observed and expected)
                benford_divergence = np.mean((observed_probs - expected_probs) ** 2) * 1000  # Scale up for readability
            else:
                benford_divergence = 0.0

            metrics = {
                "periodicity_score": round(float(periodicity_score), 4),
                "sparsity_score": round(float(sparsity_score), 4),
                "hf_energy_ratio": round(float(hf_ratio), 4),
                "benford_divergence": round(float(benford_divergence), 4)
            }

            plt.figure(figsize=(10, 5))
            plt.hist(dct_coefficients, bins=200, range=(-20, 20), color='purple', alpha=0.7, log=True)

            title_str = (f"Periodicity: {metrics['periodicity_score']} | "
                         f"Benford Div: {metrics['benford_divergence']}")
            plt.title(f'DCT Analysis\n{title_str}')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Count (Log Scale)')
            plt.grid(True, alpha=0.3)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            return metrics, buf

        except Exception as e:
            print(f"Error performing DCT Analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, None


if __name__ == "__main__":
    input_path = "Testing Images/testcase4.jpg"
    start_time = time.time()

    dct_metrics, dct_graph = DCT.dct_analyze(input_path)

    end_time = time.time()
    print(f"DCT Test took {end_time - start_time:.4f} seconds")
    print(dct_metrics)

    if dct_graph:
        with open("Results/dct_result.png", "wb") as f:
            f.write(dct_graph.getbuffer())