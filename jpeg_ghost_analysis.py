from dqt_aware_ela_test import ELA
import time

class GHOST:
    @staticmethod
    def jpeg_ghost(img_path):
        metrics, image = ELA.ela(img_path, quality=50)
        return metrics, image

if __name__ == "__main__":
    input_path = "twitter2.jpg" # Change this to image you want to check
    start_time = time.time()
    jpeg_ghost_result_metrics, jpeg_ghost_result_image = GHOST.jpeg_ghost(input_path)
    end_time = time.time()
    duration = end_time - start_time
    print(f"JPEG Ghost Analysis took {duration:.4f} seconds")

    print(jpeg_ghost_result_metrics)

    if jpeg_ghost_result_image:
        jpeg_ghost_result_image.save("jpeg_ghost_result.png")
