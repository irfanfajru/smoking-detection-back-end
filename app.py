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
    # inference
    # detection for custom weight
    # modelDetect(f"storage/upload/{filename}",
    #             weights=["weights/yolov7custom.pt"])
    
    # detection for yolov7 ms coco weight
    # only for person class
    modelDetect(f"storage/upload/{filename}",
                classes=[0],
                weights=["weights/yolov7.pt"])

    # detect multi model weight
    # detect_multi_model(filename)

    return {
        "success": True,
        "message": "Gambar berhasil dideteksi",
        "data": {
            "resultUrl": url_for("result", filename=filename, _external=True),
        }
    }

# get result


@ app.get("/result/<filename>")
def result(filename):
    return send_file(f"storage/result/{secure_filename(filename)}")
