import os
import io
import re
import time
from PIL import Image, ImageCms
import pyexiv2


class Metadata:
    @staticmethod
    def analyze(image_path):
        # Initialize the report structure
        report = {
            "flags": {
                "is_suspicious_hex": False,
                "thumbnail_mismatch": False,
                "high_chroma_sampling": False,  # e.g. 4:4:4
                "deep_edit_history": False,
                "metadata_stripped": True,
                "resolution_mismatch": False,
                "timestamp_mismatch": False,
                "software_trace_found": False,
                "has_geo_tag": False,
                "valid_soi": False,
                "valid_eoi": False,
                "has_icc_profile": False,
                "has_APP0": False,
                "has_APP1": False
            },
            "data": {
                "segment_details": {},
                "hex_signatures": [],
                "history_count": 0,
                "thumbnail_analysis": "Not Detected",
                "camera_make": None,
                "camera_model": None,
                "software_trace": None,
                "icc_profile_name": None,
                "chroma_subsampling": None,
                "actual_dims": (0, 0)
            }
        }

        try:
            # Structure Analysis (SOI/EOI)
            try:
                with open(image_path, 'rb') as f:
                    soi = f.read(2)
                    report['flags']['valid_soi'] = (soi == b'\xFF\xD8')  # checks Start of Image marker
                    f.seek(-2, 2)
                    eoi = f.read(2)
                    report['flags']['valid_eoi'] = (eoi == b'\xFF\xD9')  # checks End of Image marker
            except Exception:
                pass

            # Segments & Chroma Analysis
            try:
                with Image.open(image_path) as img:
                    report['data']['actual_dims'] = img.size  # captures actual pixel dimensions

                    # Check Chroma Subsampling
                    subsampling = getattr(img, 'jpeg_subsampling', None)
                    if subsampling == 0 or subsampling == '4:4:4':
                        report['flags']['high_chroma_sampling'] = True
                        report['data']['chroma_subsampling'] = "4:4:4 (High/Synthetic)"
                    else:
                        report['data']['chroma_subsampling'] = f"4:2:0 or other ({subsampling})"

                    # Loop through APP Markers (Segment Analysis)
                    if hasattr(img, 'applist'):
                        for marker_name, data in img.applist:
                            report['flags'][f"has_{marker_name}"] = True
                            if marker_name not in report['data']['segment_details']:
                                report['data']['segment_details'][marker_name] = []
                            report['data']['segment_details'][marker_name].append(len(data))

                    # ICC Profile Analysis
                    raw_icc = img.info.get('icc_profile')
                    if raw_icc:
                        report['flags']['has_icc_profile'] = True
                        try:
                            p = ImageCms.getOpenProfile(io.BytesIO(raw_icc))
                            desc = p.profile.profile_description
                            report['data']['icc_profile_name'] = desc
                        except:
                            pass
            except Exception as e:
                print(f"Segment Analysis Error: {e}")

            # Metadata Tags & History Analysis
            try:
                metadata = pyexiv2.ImageMetadata(image_path)
                metadata.read()

                # Helper to safely get raw values
                def get_val(k):
                    try:
                        return metadata[k].raw_value
                    except:
                        return None

                # Check if metadata exists
                if metadata.exif_keys: report['flags']['metadata_stripped'] = False
                if metadata.xmp_keys: report['flags']['metadata_stripped'] = False

                # Timestamp Consistency Check
                dt_orig = get_val('Exif.Photo.DateTimeOriginal')
                dt_digi = get_val('Exif.Photo.DateTimeDigitized')
                if dt_orig and dt_digi and dt_orig != dt_digi:
                    report['flags']['timestamp_mismatch'] = True

                # Resolution Consistency Check
                mw = get_val('Exif.Photo.PixelXDimension')
                mh = get_val('Exif.Photo.PixelYDimension')
                aw, ah = report['data'].get('actual_dims', (0, 0))
                if mw and mh and (int(mw) != aw or int(mh) != ah):
                    report['flags']['resolution_mismatch'] = True

                # Geo Tag Check
                if 'Exif.GPSInfo.GPSLatitude' in metadata.exif_keys:
                    report['flags']['has_geo_tag'] = True

                # Software Trace Check
                soft = get_val('Exif.Image.Software') or get_val('Xmp.xmp.CreatorTool')
                if soft:
                    report['flags']['software_trace_found'] = True
                    report['data']['software_trace'] = soft

                make = get_val('Exif.Image.Make')
                if make:
                    report['data']['camera_make'] = make

                model = get_val('Exif.Image.Model')
                if make:
                    report['data']['camera_model'] = model

                # XMP History Count (Deep Edit Check)
                history_keys = [k for k in metadata.xmp_keys if 'History' in k]
                count = len(history_keys)
                report['data']['history_count'] = count
                if count > 5:
                    report['flags']['deep_edit_history'] = True

            except Exception:
                pass  # If metadata fails (e.g. strict format issues), continue

            # Thumbnail Carving & Hex Signatures
            try:
                with open(image_path, 'rb') as f:
                    data = f.read()  # read entire binary once for both operations

                # Hex Signature Scan
                signatures = [
                    b'ComfyUI', b'Automatic1111', b'Stable Diffusion',
                    b'Midjourney', b'DALL-E', b'Adobe Firefly',
                    b'Photoshop', b'GIMP', b'Creator: Canva'
                ]
                found_sigs = []
                for sig in signatures:
                    if sig in data:
                        found_sigs.append(sig.decode('utf-8', errors='ignore'))

                if found_sigs:
                    report['flags']['is_suspicious_hex'] = True
                    report['data']['hex_signatures'] = list(set(found_sigs))

                # Thumbnail Carving (Brute Force)
                offsets = [m.start() for m in re.finditer(b'\xFF\xD8', data)]

                if len(offsets) > 1:
                    thumb_offset = offsets[1]  # Use second SOI as likely thumbnail start
                    report['data'][
                        'thumbnail_debug'] = f"Found {len(offsets)} JPEG headers. Testing offset {thumb_offset}..."

                    try:
                        with io.BytesIO(data[thumb_offset:]) as thumb_io:
                            with Image.open(thumb_io) as thumb:
                                thumb.load()
                                t_w, t_h = thumb.size
                                m_w, m_h = report['data'].get('actual_dims', (0, 0))

                                if m_w > 0:
                                    main_ar = m_w / m_h
                                    thumb_ar = t_w / t_h
                                    # Compare Aspect Ratios
                                    if abs(main_ar - thumb_ar) > 0.1:
                                        report['flags']['thumbnail_mismatch'] = True
                                        report['data'][
                                            'thumbnail_analysis'] = f"MISMATCH: Main AR {main_ar:.2f} vs Thumb AR {thumb_ar:.2f}"
                                    else:
                                        report['data']['thumbnail_analysis'] = f"Consistent ({t_w}x{t_h})"
                                else:
                                    report['data'][
                                        'thumbnail_analysis'] = f"Found Thumbnail ({t_w}x{t_h}) - Main dims missing"
                    except Exception as e:
                        report['data']['thumbnail_analysis'] = f"Carving Failed: {e}"
                else:
                    report['data']['thumbnail_analysis'] = "No Embedded Thumbnail Found (Binary Scan)"

            except Exception as e:
                report['data']['thumbnail_analysis'] = f"Analysis Error: {e}"

            return report

        except Exception as e:
            print(f"Error performing Metadata Analysis: {e}")
            return None


if __name__ == "__main__":
    input_path = "/home/pri/Dataset/Real/Camera Dataset/Dresden_Exp/Nikon_D70/Nikon_D70_0_19607.JPG"  # Change this to image you want to check

    if os.path.exists(input_path):
        start_time = time.time()

        metadata_report = Metadata.analyze(input_path)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Metadata Test took {duration:.4f} seconds")
        print(metadata_report)
    else:
        print("Image path not found.")