import streamlit as st
import ollama
import numpy as np

st.title("📚 Simple RAG Chatbot")

st.write("Ask a question about your notes.")

# Read notes
with open("notes.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Split notes into chunks
chunks = [
    x.strip()
    for x in document.split("\n\n")
    if x.strip()
]
if not chunks:
    st.error("notes.txt is empty. Please add some text to notes.txt.")
    st.stop()
# Create embeddings
embeddings = []

for chunk in chunks:

    result = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    # Check whether Ollama returned an embedding
    if not result["embeddings"]:
        st.error("Ollama did not return an embedding.")
        st.stop()

    embedding = result["embeddings"][0]

    embeddings.append(embedding)


# Question box
question = st.text_input("Enter your question")


# Ask button
if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        # Create embedding for question
        result = ollama.embed(
            model="nomic-embed-text",
            input=question
        )

        if not result["embeddings"]:
            st.error("Ollama did not create an embedding.")
            st.stop()

        question_embedding = np.array(
            result["embeddings"][0]
        )

        # Find most relevant chunk
        best_chunk = chunks[0]
        best_score = -1

        for i in range(len(chunks)):

            chunk_embedding = np.array(
                embeddings[i]
            )

            score = np.dot(
                question_embedding,
                chunk_embedding
            ) / (
                np.linalg.norm(question_embedding)
                * np.linalg.norm(chunk_embedding)
            )

            if score > best_score:
                best_score = score
                best_chunk = chunks[i]

        # Send retrieved information to Ollama
        prompt = f"""
Answer the question using only the information below.

Information:
{best_chunk}

Question:
{question}

If the answer is not in the information,
say "I don't know."
"""

        response = ollama.chat(
            model="gemma4",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        st.subheader("🤖 Answer")

        st.write(
            response["message"]["content"]
        )