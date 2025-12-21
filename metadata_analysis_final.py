import io
import os
from pyexiv2 import ImageMetadata
from PIL import Image, ImageCms
import time


class MetadataForensics:
    def __init__(self, image_path):
        self.image_path = image_path
        self.report = {
            "flags": {
                "is_suspicious_hex": False,
                "thumbnail_mismatch": False,
                "high_chroma_sampling": False,  # e.g. 4:4:4
                "deep_edit_history": False,
                "metadata_stripped": True,
                "resolution_mismatch": False,
                "timestamp_mismatch": False,
                "software_trace_found": False,
                "has_geo_tag": False
            },
            "data": {
                "segment_details": {},
                "hex_signatures": [],
                "history_count": 0,
                "thumbnail_analysis": "Not Detected",
                "software_trace": None,
                "icc_profile_name": None,
                "chroma_subsampling": None
            }
        }

    def analyze_structure(self):
        """Checks file integrity (SOI/EOI)"""
        try:
            with open(self.image_path, 'rb') as f:
                soi = f.read(2)
                self.report['flags']['valid_soi'] = (soi == b'\xFF\xD8')
                f.seek(-2, 2)
                eoi = f.read(2)
                self.report['flags']['valid_eoi'] = (eoi == b'\xFF\xD9')
        except Exception:
            self.report['flags']['valid_soi'] = False
            self.report['flags']['valid_eoi'] = False

    def analyze_segments_and_chroma(self):
        """
        Scans APPn segments, ICC Profiles, and Chroma Subsampling.
        """
        try:
            with Image.open(self.image_path) as img:
                self.report['data']['actual_dims'] = img.size

                # Check Chroma Subsampling (SOF0)
                subsampling = getattr(img, 'jpeg_subsampling', None)
                if subsampling == 0 or subsampling == '4:4:4':
                    self.report['flags']['high_chroma_sampling'] = True
                    self.report['data']['chroma_subsampling'] = "4:4:4 (High/Synthetic)"
                else:
                    self.report['data']['chroma_subsampling'] = f"4:2:0 or other ({subsampling})"

                # Loop APP Markers
                if hasattr(img, 'applist'):
                    for marker_name, data in img.applist:
                        self.report['flags'][f"has_{marker_name}"] = True
                        if marker_name not in self.report['data']['segment_details']:
                            self.report['data']['segment_details'][marker_name] = []
                        self.report['data']['segment_details'][marker_name].append(len(data))

                # ICC Profile
                raw_icc = img.info.get('icc_profile')
                if raw_icc:
                    self.report['flags']['has_icc_profile'] = True
                    try:
                        p = ImageCms.getOpenProfile(io.BytesIO(raw_icc))
                        desc = p.profile.profile_description
                        self.report['data']['icc_profile_name'] = desc
                    except:
                        pass
        except Exception as e:
            print(f"Segment Error: {e}")

    def analyze_tags(self):
        """Parses Metadata Tags for inconsistencies"""
        try:
            metadata = ImageMetadata(self.image_path)
            metadata.read()

            # Helper
            def get_val(k):
                try:
                    return metadata[k].raw_value
                except:
                    return None

            # Flags for existence
            if metadata.exif_keys: self.report['flags']['metadata_stripped'] = False
            if metadata.xmp_keys: self.report['flags']['metadata_stripped'] = False

            # Timestamp Check
            dt_orig = get_val('Exif.Photo.DateTimeOriginal')
            dt_digi = get_val('Exif.Photo.DateTimeDigitized')
            if dt_orig and dt_digi and dt_orig != dt_digi:
                self.report['flags']['timestamp_mismatch'] = True

            # Resolution Check
            mw = get_val('Exif.Photo.PixelXDimension')
            mh = get_val('Exif.Photo.PixelYDimension')
            aw, ah = self.report['data'].get('actual_dims', (0, 0))
            if mw and mh and (int(mw) != aw or int(mh) != ah):
                self.report['flags']['resolution_mismatch'] = True

            # Software/Geo
            if 'Exif.GPSInfo.GPSLatitude' in metadata.exif_keys:
                self.report['flags']['has_geo_tag'] = True

            soft = get_val('Exif.Image.Software') or get_val('Xmp.xmp.CreatorTool')
            if soft:
                self.report['flags']['software_trace_found'] = True
                self.report['data']['software_trace'] = soft

        except Exception as e:
            print(f"Tag Parse Error: {e}")

    def analyze_thumbnail(self):
        """
        Brute Force Carving: Scans the raw file bytes for hidden JPEG headers (FF D8).
        This ignores broken metadata pointers and finds the thumbnail physically.
        """
        self.report['data']['thumbnail_analysis'] = "Not Detected"

        try:
            with open(self.image_path, 'rb') as f:
                data = f.read()

            import re
            offsets = [m.start() for m in re.finditer(b'\xFF\xD8', data)]

            if len(offsets) > 1:
                thumb_offset = offsets[1]

                # Check consistency
                self.report['data'][
                    'thumbnail_debug'] = f"Found {len(offsets)} JPEG headers. Testing offset {thumb_offset}..."

                try:
                    with io.BytesIO(data[thumb_offset:]) as thumb_io:
                        with Image.open(thumb_io) as thumb:
                            thumb.load()
                            t_w, t_h = thumb.size

                            m_w, m_h = self.report['data'].get('actual_dims', (0, 0))

                            if m_w > 0:
                                main_ar = m_w / m_h
                                thumb_ar = t_w / t_h

                                # Check Aspect Ratio
                                if abs(main_ar - thumb_ar) > 0.1:
                                    self.report['flags']['thumbnail_mismatch'] = True
                                    self.report['data'][
                                        'thumbnail_analysis'] = f"MISMATCH: Main AR {main_ar:.2f} vs Thumb AR {thumb_ar:.2f}"
                                else:
                                    self.report['data']['thumbnail_analysis'] = f"Consistent ({t_w}x{t_h})"
                            else:
                                self.report['data'][
                                    'thumbnail_analysis'] = f"Found Thumbnail ({t_w}x{t_h}) - Main dims missing"

                except Exception as e:
                    self.report['data'][
                        'thumbnail_analysis'] = f"Carving Failed: Found header but could not decode ({e})"

            else:
                self.report['data']['thumbnail_analysis'] = "No Embedded Thumbnail Found (Binary Scan)"

        except Exception as e:
            self.report['data']['thumbnail_analysis'] = f"Carving Error: {e}"
        except Exception as e:
            self.report['data']['thumbnail_analysis'] = f"CRITICAL ERROR: {str(e)}"

    def analyze_hex_signatures(self):
        """
        Scans the raw binary for known AI or Editing signatures.
        """
        signatures = [
            b'ComfyUI', b'Automatic1111', b'Stable Diffusion',
            b'Midjourney', b'DALL-E', b'Adobe Firefly',
            b'Photoshop', b'GIMP', b'Creator: Canva'
        ]

        found_sigs = []
        try:
            with open(self.image_path, 'rb') as f:
                content = f.read()
                for sig in signatures:
                    if sig in content:
                        found_sigs.append(sig.decode('utf-8', errors='ignore'))
        except Exception:
            pass

        if found_sigs:
            self.report['flags']['is_suspicious_hex'] = True
            self.report['data']['hex_signatures'] = list(set(found_sigs))

    def analyze_xmp_history(self):
        """
        Counts the number of history states in XMP.
        """
        try:
            metadata = ImageMetadata(self.image_path)
            metadata.read()

            history_keys = [k for k in metadata.xmp_keys if 'History' in k]
            count = len(history_keys)

            self.report['data']['history_count'] = count

            if count > 5:
                self.report['flags']['deep_edit_history'] = True

        except Exception:
            pass

    def run_test(self):
        self.analyze_structure()
        self.analyze_segments_and_chroma()
        self.analyze_tags()
        self.analyze_thumbnail()
        self.analyze_hex_signatures()
        self.analyze_xmp_history()
        return self.report


image_path = "Testing Images/testcase3.jpg"  # Change this to image you want to check

if os.path.exists(image_path):
    start_time = time.time()
    forensics = MetadataForensics(image_path)
    end_time = time.time()
    results = forensics.run_test()

    '''print("\n[ Flags ]")
    for k, v in results['flags'].items():
        status = "[!]" if v else "[ ]"
        print(f" {status} {k}")

    print("\n[ Data ]")
    for k, v in results['data'].items():
        print(f" {k}: {v}")

    duration = end_time - start_time
    print(f"Metadata Analysis took {duration:.4f} seconds")
else:
    print("Image not found.")'''