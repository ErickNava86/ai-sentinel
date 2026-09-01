from flask import Flask, render_template, request
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
client = OpenAI()



@app.route("/", methods=["GET", "POST"])
def index():
    response = None

    if request.method == "POST":
        user_message = request.form.get("message")

        try:
            ai_resonse = client.responses.create(
                model="gpt-5.6-luna",
                input=user_message
            )

            response = ai_resonse.output_text

        except Exception as error:
            print(f"OpenAI API error: {error}")
            response="Sentinel is temporarily unavailable. Please try again later."

    return render_template("index.html", response=response)


if __name__ == "__main__":
    app.run(debug=True)