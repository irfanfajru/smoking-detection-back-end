from flask import Flask, request, send_file, url_for
from markupsafe import escape
from werkzeug.utils import secure_filename
import uuid
app = Flask(__name__)


@app.route("/")
def index():
    return "<p>Hello Dek, mau ngapain ?</p>"

# post detection image


@app.post("/detect")
def detect():
    file = request.files['image']
    filename = f"{str(uuid.uuid4())}{secure_filename(file.filename)}"
    file.save(f"storage/{filename}")
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
    return send_file(f"storage/{secure_filename(filename)}")
