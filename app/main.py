"""
Flask app for CT Scan model
"""

from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home_page():
    if request.method == "POST":
        # desc = request.form[""]
        return render_template("base.html")
    else:
        return render_template("base.html")

if __name__ == "__main__":
    app.run(debug=True)