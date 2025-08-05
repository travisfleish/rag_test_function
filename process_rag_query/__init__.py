import azure.functions as func
import logging
import json
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Stubbed Retriever Agent (mock)
def retrieve_context(query: str, top_k: int = 3) -> list:
    # Replace this with actual Azure Cognitive Search logic later
    return [
        "Clause: Payment terms are net-30 from invoice date.",
        "Clause: Client agrees to pay within 30 days of receipt of invoice.",
        "Clause: Late payments incur a 5% fee after 30 days."
    ][:top_k]

# Synthesizer Agent using OpenAI
def synthesize_answer(question: str, context: list) -> str:
    joined_context = "\n\n".join(context)
    prompt = f"""You are a helpful AI assistant. Use the context below to answer the user's question.

Context:
{joined_context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🧠 Processing RAG query...")

    try:
        req_body = req.get_json()
        question = req_body.get("question", "")
        top_k = int(req_body.get("top_k", 3))

        if not question:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'question' in request body."}),
                status_code=400,
                mimetype="application/json"
            )

        # Run agents
        context = retrieve_context(question, top_k)
        answer = synthesize_answer(question, context)

        return func.HttpResponse(
            json.dumps({
                "question": question,
                "context_used": context,
                "answer": answer
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("❌ RAG processing failed.")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
