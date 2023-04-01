from flask import Flask, request, send_file, url_for
from markupsafe import escape
from werkzeug.utils import secure_filename
import uuid
import command
from detect import detect as modelDetect
app = Flask(__name__)


@app.route("/")
def index():
    return "<p>Hello Dek, mau ngapain ?</p>"

# post detection image


@app.post("/detect")
def detect():
    file = request.files['image']
    filename = f"{str(uuid.uuid4())}{secure_filename(file.filename)}"
    file.save(f"storage/upload/{filename}")
    modelDetect(f"storage/upload/{filename}")
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
