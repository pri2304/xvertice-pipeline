import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io


def ela(image_path, quality=90):
    try:
        raw_image = Image.open(image_path)
        quantization_tables = getattr(raw_image, 'quantization', None)

        original_image = raw_image.convert('RGB')

        buffer = io.BytesIO()
        if quantization_tables:
            original_image.save(buffer, 'JPEG', qtables=quantization_tables)
        else:
            original_image.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer).convert('RGB')

        ela_image = ImageChops.difference(original_image, compressed_image)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255 / max_diff if max_diff != 0 else 1

        ela_array = np.array(ela_image, dtype=np.int16)

        per_pixel_max = ela_array.max(axis=2).astype(np.float32)

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        floor = max(5, np.percentile(per_pixel_max, 10))
        content_mask = per_pixel_max > floor
        thr = np.percentile(per_pixel_max[content_mask], 98)


        mean_intensity = float(np.mean(per_pixel_max))
        std_dev = float(np.std(per_pixel_max))
        if np.sum(content_mask) > 0:
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
    input_path = "testcase3.jpg"

    ela_result_metrics, ela_result_image = ela(input_path)

    print(ela_result_metrics)

    if ela_result_image:
        ela_result_image.save("ela_result.png")