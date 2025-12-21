import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import scipy.ndimage
import time


class NoiseAnalysis:

    @staticmethod
    def _normalize_to_image(data_array):
        max_val = np.max(np.abs(data_array))
        scale = 255.0 / max_val if max_val != 0 else 1.0
        scaled_data = (np.abs(data_array) * scale).clip(0, 255).astype(np.uint8)
        return Image.fromarray(scaled_data, mode='L')

    @staticmethod
    def _draw_histogram(counts, title="Grid"):
        """Helper to draw a simple graph of the 8x8 grid for visualization."""
        w, h = 256, 150
        img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Normalize counts to fit height
        max_val = max(counts) if max(counts) > 0 else 1
        min_val = min(counts)
        range_val = max_val - min_val if max_val != min_val else 1

        bar_width = w // 8

        # Draw bars for the 8 positions (0-7)
        for i, val in enumerate(counts):
            # Invert y because 0 is top
            height_norm = int(((val - min_val) / range_val) * (h - 20))
            x0 = i * bar_width + 10
            y0 = h - 10
            x1 = x0 + bar_width - 5
            y1 = h - 10 - height_norm

            # Highlight the '0' index (expected grid start) in Red, others in Blue
            color = (255, 0, 0) if i == 0 else (0, 0, 255)
            draw.rectangle([x0, y1, x1, y0], fill=color)
            draw.text((x0 + 5, h - 25), str(i), fill=(0, 0, 0))  # Index label

        draw.text((5, 5), title, fill=(0, 0, 0))
        return img

    @staticmethod
    def exposure_check(image_path):
        """
        NEW: Checks if image is valid for noise analysis.
        Returns flags to SKIP noise tests if image is pitch black or flat gray.
        """
        try:
            with Image.open(image_path) as img:
                gray = np.array(img.convert('L'))

                is_too_dark = np.mean(gray) < 20

                is_flat = np.std(gray) < 10

                return {
                    "is_too_dark": bool(is_too_dark),
                    "is_flat": bool(is_flat),
                    "mean_brightness": float(np.mean(gray)),
                    "global_std": float(np.std(gray))
                }
        except Exception as e:
            print(f"Error in exposure check: {e}")
            return {"is_too_dark": False, "is_flat": False}

    @staticmethod
    def laplacian_check(image_path):
        try:
            with Image.open(image_path) as img:
                gray_image = img.convert('L')
                img_array = np.array(gray_image, dtype=np.float32)

            laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            residual_map = scipy.ndimage.convolve(img_array, laplacian_kernel)
            abs_map = np.abs(residual_map)

            mean_intensity = float(np.mean(abs_map))
            std_dev = float(np.std(abs_map))
            threshold = np.percentile(abs_map, 95)
            bright_ratio = float(np.sum(abs_map > threshold) / abs_map.size * 100)

            metrics = {
                "test_type": "Laplacian Noise",
                "mean_intensity": mean_intensity,
                "standard_deviation": std_dev,
                "bright_ratio": bright_ratio
            }

            result_image = NoiseAnalysis._normalize_to_image(residual_map)
            result_image = ImageEnhance.Brightness(result_image).enhance(5.0)

            return metrics, result_image

        except Exception as e:
            print(f"Error performing Laplacian Analysis: {e}")
            return None, None

    @staticmethod
    def median_check(image_path, filter_size=3):
        try:
            with Image.open(image_path) as img:
                gray_image = img.convert('L')
                original_array = np.array(gray_image, dtype=np.float32)

            median_array = scipy.ndimage.median_filter(original_array, size=filter_size)
            residual_map = original_array - median_array
            abs_map = np.abs(residual_map)

            mean_intensity = float(np.mean(abs_map))
            std_dev = float(np.std(abs_map))
            threshold = np.percentile(abs_map, 98)
            bright_ratio = float(np.sum(abs_map > threshold) / abs_map.size * 100)

            metrics = {
                "test_type": "Median Residual",
                "mean_intensity": mean_intensity,
                "standard_deviation": std_dev,
                "bright_ratio": bright_ratio
            }

            result_image = NoiseAnalysis._normalize_to_image(residual_map)
            result_image = ImageEnhance.Brightness(result_image).enhance(10.0)

            return metrics, result_image

        except Exception as e:
            print(f"Error performing Median Analysis: {e}")
            return None, None

    @staticmethod
    def local_variance_check(image_path, window_size=3):
        """
        Calculates Local Variance.
        Updated: Uses Absolute Threshold (< 5.0) to correctly identify smooth AI textures.
        """
        try:
            with Image.open(image_path) as img:
                gray_image = img.convert('L')
                img_array = np.array(gray_image, dtype=np.float32)

            # Calculation: Var(X) = E[X^2] - (E[X])^2
            mean = scipy.ndimage.uniform_filter(img_array, window_size)
            sqr_mean = scipy.ndimage.uniform_filter(img_array ** 2, window_size)
            variance_map = sqr_mean - mean ** 2

            # Clean up potential negative values from float precision errors
            variance_map = np.maximum(variance_map, 0)

            # Metrics
            mean_var = float(np.mean(variance_map))
            max_var = float(np.max(variance_map))

            smooth_pixels = np.sum(variance_map < 5.0)
            smooth_ratio = float(smooth_pixels / variance_map.size * 100)

            metrics = {
                "test_type": "Local Variance",
                "average_variance": mean_var,
                "max_variance": max_var,
                "smooth_ratio": smooth_ratio
            }

            # Visual: Standard normalization
            result_image = NoiseAnalysis._normalize_to_image(variance_map)
            result_image = ImageEnhance.Contrast(result_image).enhance(2.0)

            return metrics, result_image

        except Exception as e:
            print(f"Error performing Local Variance Analysis: {e}")
            return None, None

    @staticmethod
    def block_boundary_check(image_path):
        """
        Analyzes the JPEG 8x8 Block Artifact Grid (BAG).
        """
        try:
            with Image.open(image_path) as img:
                gray_image = img.convert('L')
                img_array = np.array(gray_image, dtype=np.float32)

            h, w = img_array.shape

            # Horizontal Grid Analysis
            row_diffs = np.sum(np.abs(img_array[:, :-1] - img_array[:, 1:]), axis=0)
            h_grid = np.zeros(8)
            for i in range(len(row_diffs)):
                h_grid[i % 8] += row_diffs[i]

            # Vertical Grid Analysis
            col_diffs = np.sum(np.abs(img_array[:-1, :] - img_array[1:, :]), axis=1)
            v_grid = np.zeros(8)
            for i in range(len(col_diffs)):
                v_grid[i % 8] += col_diffs[i]

            # Calculate "Bag Error"
            def get_bag_metric(grid_arr):
                min_val = np.min(grid_arr)
                avg_val = np.mean(grid_arr)
                if avg_val == 0: return 0.0
                return float((avg_val - min_val) / avg_val)

            h_score = get_bag_metric(h_grid)
            v_score = get_bag_metric(v_grid)

            # Detect Crop
            h_shift = int(np.argmin(h_grid))
            v_shift = int(np.argmin(v_grid))

            metrics = {
                "test_type": "Block Boundary (BAG)",
                "horizontal_grid_strength": h_score,
                "vertical_grid_strength": v_score,
                "crop_shift_x": h_shift,
                "crop_shift_y": v_shift,
                "is_aligned": (h_shift == 0 and v_shift == 0)
            }

            # Visual
            graph_h = NoiseAnalysis._draw_histogram(h_grid, title=f"Horizontal (Shift: {h_shift})")
            graph_v = NoiseAnalysis._draw_histogram(v_grid, title=f"Vertical (Shift: {v_shift})")
            result_image = Image.new('RGB', (256, 300))
            result_image.paste(graph_h, (0, 0))
            result_image.paste(graph_v, (0, 150))

            return metrics, result_image

        except Exception as e:
            print(f"Error performing BAG Analysis: {e}")
            return None, None


if __name__ == "__main__":
    input_path = "Testing Images/testcase13.jpg"  # Change this to image you want to check

    # 0. Exposure Check
    start_time = time.time()
    exposure = NoiseAnalysis.exposure_check(input_path)
    end_time = time.time()
    print("Exposure Check:", exposure)
    duration = end_time - start_time
    print(f"Exposure Check took {duration:.4f} seconds")

    # 1. Laplacian
    start_time = time.time()
    lap_metrics, lap_image = NoiseAnalysis.laplacian_check(input_path)
    end_time = time.time()
    print("Laplacian Metrics:", lap_metrics)
    if lap_image: lap_image.save("Results/laplacian_result.png")
    duration = end_time - start_time
    print(f"Laplacian took {duration:.4f} seconds")

    # 2. Median
    start_time = time.time()
    med_metrics, med_image = NoiseAnalysis.median_check(input_path)
    end_time = time.time()
    print("Median Metrics:", med_metrics)
    if med_image: med_image.save("Results/median_result.png")
    duration = end_time - start_time
    print(f"Median took {duration:.4f} seconds")

    # 3. Local Variance
    start_time = time.time()
    var_metrics, var_image = NoiseAnalysis.local_variance_check(input_path)
    end_time = time.time()
    print("Variance Metrics:", var_metrics)
    if var_image: var_image.save("Results/variance_result.png")
    duration = end_time - start_time
    print(f"Median took {duration:.4f} seconds")

    # 4. Block Boundary
    start_time = time.time()
    bag_metrics, bag_image = NoiseAnalysis.block_boundary_check(input_path)
    end_time = time.time()
    print("BAG Metrics:", bag_metrics)
    if bag_image: bag_image.save("Results/bag_result.png")
    duration = end_time - start_time
    print(f"Median took {duration:.4f} seconds")