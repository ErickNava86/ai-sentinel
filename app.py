from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    response = None

    if request.method == "POST":
        user_message = request.form.get("message")

        response = f"Sentinel received: {user_message}"

    return render_template("index.html", response=response)


if __name__ == "__main__":
    app.run(debug=True)