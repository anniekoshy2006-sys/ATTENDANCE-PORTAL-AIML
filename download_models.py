"""
download_models.py - fetch the two small neural-network files used for accurate recognition.

    python download_models.py

If your network blocks GitHub, download these two files in a browser and put them in ./models :
  face_detection_yunet_2023mar.onnx   (about 0.2 MB)
  face_recognition_sface_2021dec.onnx (about 37 MB)
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
"""
import sys
from face_engine import ensure_sface_models, YUNET_PATH, SFACE_PATH

if __name__ == "__main__":
    print("Downloading face models into ./models ...")
    if ensure_sface_models():
        print("Done.  Now run:  python train.py   (then)   python app.py")
    else:
        print(__doc__)
        sys.exit(1)
