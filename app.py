from flask import Flask, request
from markupsafe import escape
app = Flask(__name__)


@app.route("/")
def index():
    return "<p>Hello world</p>"


# route with parameter


@app.route("/<name>")
def name(name):
    return f"<p>hello {escape(name)}</p>"


@app.post("/detect")
def detect():
    return "ini hasil deteksi anda"
