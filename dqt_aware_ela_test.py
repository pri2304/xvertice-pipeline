import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import time

class ELA:
    @staticmethod
    def ela(image_path, quality=None):
        try:
            raw_image = Image.open(image_path)
            quantization_tables = getattr(raw_image, 'quantization', None) #fetches image's existing q-tables

            original_image = raw_image.convert('RGB')

            buffer = io.BytesIO()
            if quality is None:
                if quantization_tables:
                    original_image.save(buffer, 'JPEG', qtables=quantization_tables) #use quantization table for quality
                else:
                    original_image.save(buffer, 'JPEG', quality=90) #if no q-table not found, use default quality for recompression
            else:
                original_image.save(buffer, 'JPEG', quality=quality)
            buffer.seek(0)
            compressed_image = Image.open(buffer).convert('RGB')

            ela_image = ImageChops.difference(original_image, compressed_image) #difference of image pixels

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            scale = 255 / max_diff if max_diff != 0 else 1

            ela_array = np.array(ela_image, dtype=np.int16) #converts to array for numerical metrics before image scaling

            per_pixel_max = ela_array.max(axis=2).astype(np.float32) #gets max pixel values from the 3 color channels

            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale) #scales image brightness for visual output

            floor = max(5, np.percentile(per_pixel_max, 10)) #floor value for pixel to be considered bright, max between 5 or value at 10th percentile
            content_mask = per_pixel_max > floor #all pixel values greater than floor = True, below = False

            mean_intensity = float(np.mean(per_pixel_max))
            std_dev = float(np.std(per_pixel_max))
            if np.sum(content_mask) > 0:
                thr = np.percentile(per_pixel_max[content_mask],98)  # threshold for brightness is all pixel values above 98th percentile
                bright_ratio = float(np.sum(per_pixel_max[content_mask] > thr) / np.sum(content_mask)*100)
            else:
                bright_ratio = 0.0

            metrics ={"mean_intensity" : mean_intensity,
                      "standard_deviation" : std_dev,
                      "bright_ratio" : bright_ratio}

            return metrics, ela_image
        except Exception as e:
            print(f"Error performing ELA Analysis: {e}")
            return None

if __name__ == "__main__":
    input_path = "twitter2.jpg" # Change this to image you want to check
    start_time = time.time()
    ela_result_metrics, ela_result_image = ELA.ela(input_path)
    end_time = time.time()
    duration = end_time - start_time
    print(f"DQT Aware ELA Test took {duration:.4f} seconds")

    print(ela_result_metrics)

    if ela_result_image:
        ela_result_image.save("ela_result.png")