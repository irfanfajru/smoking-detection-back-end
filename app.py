from flask import Flask, request, send_file, url_for
from markupsafe import escape
from werkzeug.utils import secure_filename
import uuid
from detect import detect as modelDetect
from flask_cors import CORS
import urllib.request
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return "<p>Hello World :)</p>"

# post detection image
@app.post("/detect")
def detect():
    file = request.files['image']
    filename = f"{str(uuid.uuid4())}{secure_filename(file.filename)}"
    file.save(f"storage/upload/{filename}")
    # inference 1
    # detection for yolov7 ms coco weight
    # only for person and motorcycle class
    # inference 2 using cig_81 weight
    # to detect cigar from frame (person and motorcycle)
    resultDetection = modelDetect(f"storage/upload/{filename}",
                                  classes=[0, 3],
                                  weights=["weights/yolov7.pt", "weights/best_cigar.pt"])

    return {
        "success": True,
        "message": "Gambar berhasil dideteksi",
        "data": {
            "resultUrl": url_for("result", filename=filename, _external=True),
            "detailDetection": resultDetection
        }
    }

# get result
@ app.get("/result/<filename>")
def result(filename):
    return send_file(f"storage/result/{secure_filename(filename)}")

# post example request
@app.post("/example")
def example():
    data = request.get_json()
    filename = str(uuid.uuid4())+(data['file_url'].split("/"))[-1]
    try:
        urllib.request.urlretrieve(
            data['file_url'], f"./storage/upload/{filename}")
        print(f"Downloaded '{filename}' from '{data['file_url']}'")
        resultDetection = modelDetect(f"storage/upload/{filename}",
                                      classes=[0, 3],
                                      weights=["weights/yolov7.pt", "weights/best_cigar.pt"])

        return {
            "success": True,
            "message": "Gambar berhasil dideteksi",
            "data": {
                "resultUrl": url_for("result", filename=filename, _external=True),
                "detailDetection": resultDetection
            }
        }

    except Exception as e:
        print(
            f"Failed to download '{filename}' from '{data['file_url']}': {e}")
        return {
            "success": False,
            "message": "Failed",
            "data": None,
        }
