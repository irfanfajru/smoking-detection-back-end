from flask import Flask, request, send_file, url_for
from markupsafe import escape
from werkzeug.utils import secure_filename
import uuid
from detect import detect as modelDetect
from detect_multi_model import detect_multi_model
from flask_cors import CORS
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
                                    classes=[0,2],
                                    weights=["weights/yolov7.pt","weights/best_cigarette.pt"])

    return {
        "success": True,
        "message": "Gambar berhasil dideteksi",
        "data": {
            "resultUrl": url_for("result", filename=filename, _external=True),
            "detailDetection":resultDetection
        }
    }

# get result


@ app.get("/result/<filename>")
def result(filename):
    return send_file(f"storage/result/{secure_filename(filename)}")
