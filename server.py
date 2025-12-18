import time
import shutil
import os
import concurrent.futures
import uuid
from fastapi import FastAPI, UploadFile, File
from noise_analysis_test import NoiseAnalysis
from jpeg_ghost_analysis import GHOST
from metadata_analysis_final import MetadataForensics
from dqt_aware_ela_test import ELA

app = FastAPI()

def run_timed_test(test_name, test_func, image_path):
    """Runs a single test and measures its execution time."""
    start = time.time()
    try:
        result = test_func(image_path)
        if isinstance(result, tuple):
            data = result[0]
        else:
            data = result

        duration = time.time() - start
        return {
            "test_name": test_name,
            "status": "success",
            "data": data,
            "time_taken": round(duration, 4)
        }
    except Exception as e:
        return {
            "test_name": test_name,
            "status": "error",
            "error": str(e),
            "time_taken": round(time.time() - start, 4)
        }

def run_metadata_wrapper(image_path):
    try:
        tool = MetadataForensics(image_path)
        return tool.run_test()
    except TypeError:
        return MetadataForensics().run_test(image_path)

@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)):
    unique_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        print("hello")
        with open(unique_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        server_start_time = time.time()
        results = {}

        job_list = [("DQT aware ELA", ELA.ela),
                    ("JPEG Ghost", GHOST.jpeg_ghost),
                    ("Metadata", run_metadata_wrapper),
                    ("Exposure Check", NoiseAnalysis.exposure_check),
                    ("Laplacian", NoiseAnalysis.laplacian_check),
                    ("Median", NoiseAnalysis.median_check),
                    ("Local Variance", NoiseAnalysis.local_variance_check),
                    ("Block Boundary", NoiseAnalysis.block_boundary_check)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_timed_test, name, func, unique_filename): name for name , func in job_list}

            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                test_name = res.pop("test_name")
                results[test_name] = res

        total_duration = time.time() - server_start_time
        return {"verdict": "COMPLETE",
                "total_server_time": round(total_duration, 4),
                "results": results
                }

    except Exception as e:
        return {"verdict": "CRASH", "error": str(e)}

    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
