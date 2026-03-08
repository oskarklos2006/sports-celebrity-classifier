from flask import Flask, request, jsonify, send_from_directory
import util
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "UI"))
MODEL_IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "images"))

app = Flask(__name__, static_folder=UI_DIR)


@app.route("/player_images/<player>/<filename>")
def player_images(player, filename):
    return send_from_directory(
        os.path.join(MODEL_IMAGES_DIR, player),
        filename
    )


@app.route("/")
def home():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(UI_DIR, "style.css")


@app.route("/script.js")
def script():
    return send_from_directory(UI_DIR, "script.js")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        result = util.predict_image(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run(debug=False)
