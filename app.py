# ---------- IMPORTS --------


from flask import Flask, render_template, request
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import json


# ------- FUNCTIONS ------------


def load_activity_logs():
    try:
        with open("activity_logs.json", "r") as file:
            return json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_activity_logs():
    with open("activity_logs.json", "w") as file:
        json.dump(activity_logs, file, indent=4)

def add_activity_log(question, source, status):
    activity_logs.append({
        "time": datetime.now().strftime("%m/%d/%Y %I:%M %p"),
        "question": question,
        "source": source,
        "status": status
    })
    
    save_activity_logs()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    return dot_product / (magnitude_a * magnitude_b)

def retrieve_relevant_chunks(question, chunks, embeddings, limit=1):
    question_embedding = get_embedding(question)

    scored_chunks = []

    for chunk, chunk_embedding in zip(chunks, embeddings):
        score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        scored_chunks.append((score, chunk))

    scored_chunks.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    return [
        chunk
        for score, chunk in scored_chunks[:limit]
        if score >= 0.30
    ]


# ---------- SET UP ------------


app = Flask(__name__)

load_dotenv()
client = OpenAI()

activity_logs = load_activity_logs()


# ------------ KNOWLEDGE BASE -------------------


knowledge_path = Path("knowledge/company_policies.txt")
company_knowledge = knowledge_path.read_text(encoding="utf-8")
knowledge_chunks = company_knowledge.split("\n\n")

chunk_embeddings = [
    get_embedding(chunk)
    for chunk in knowledge_chunks
    ]


# --------------- ROUTES ------------------


@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    sources = None

    if request.method == "POST":
        user_message = request.form.get("message")
        relevant_chunks = retrieve_relevant_chunks(
            user_message,
            knowledge_chunks,
            chunk_embeddings
        )

        if not relevant_chunks:
            response = "I could not find that information in the company knowledge base."

            add_activity_log(
                user_message,
                "None",
                "No relevant knowledge"
            )

        else:
            relevant_context = "\n\n".join(relevant_chunks)
            sources = [
                chunk.splitlines()[0]
                for chunk in relevant_chunks
            ]

            try:
                ai_response = client.responses.create(
                    model="gpt-5.6-luna",
                    input=f"""
                        You are AI Sentinel, an internal company assistant.
                        Answer the employee's question using ONLY the company information
                        provided below.
                        If the answer cannot be found in the company information, say:
                        "I could not find that information in the company knowledge base."
                        COMPANY INFORMATION:
                        {relevant_context}
                        EMPLOYEE QUESTION:
                        {user_message}
                        """
                    )

                response = ai_response.output_text

                add_activity_log(
                    user_message,
                    sources,
                    "Success"
                )

            except Exception as error:
                print(f"OpenAI API error: {error}")
                response="Sentinel is temporarily unavailable. Please try again later."

                add_activity_log(
                    user_message,
                    sources,
                    "Error"
                )

    return render_template("index.html", response=response, sources=sources)


@app.route("/knowledge")
def knowledge_base():
    documents = [
        {
            "name": knowledge_path.name,
            "chunks": len(knowledge_chunks)
        }
    ]

    return render_template(
        "knowledge.html",
        documents=documents
    )


@app.route("/logs")
def activity_logs_page():
    return render_template(
        "logs.html",
        logs=activity_logs
    )


# -------------- START APPLICATION -----------------


if __name__ == "__main__":
    app.run(debug=True)

