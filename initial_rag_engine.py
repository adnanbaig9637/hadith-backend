import mysql.connector
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Step 1: Connect to MySQL
# -----------------------------
db_config = {
    "host": "localhost",
    "user": "docker",
    "password": "docker",
    "database": "hadithdb",
    "port": 3307
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor(dictionary=True)

cursor.execute("SELECT * FROM EnglishHadithTable")
rows = cursor.fetchall()
print(f"Fetched {len(rows)} rows from DB")

# -----------------------------
# Step 2: Prepare text data
# -----------------------------
documents = []
for row in rows:
    text = "\n".join([f"{k}: {v}" for k, v in row.items()])
    documents.append(text)

# -----------------------------
# Step 3: Load or compute embeddings
# -----------------------------
MODEL_PATH = "models/multi_qa_mpnet"

if os.path.exists(MODEL_PATH):
    print("Loading local embedding model...")
    embedding_model = SentenceTransformer(MODEL_PATH)
else:
    print("Downloading and saving embedding model locally...")
    embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    embedding_model.save(MODEL_PATH)

# -----------------------------
# Step 4: Load or create FAISS index
# -----------------------------
FAISS_INDEX_PATH = "faiss/hadith_index.faiss"
DOCS_PATH = "faiss/documents.npy"

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCS_PATH):
    print("Loading FAISS index and documents from disk...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    documents = np.load(DOCS_PATH, allow_pickle=True).tolist()
else:
    print("Computing embeddings and creating FAISS index...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    os.makedirs("faiss", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(DOCS_PATH, np.array(documents))
    print(f"FAISS index built with {index.ntotal} vectors")

# -----------------------------
# Step 5: Initialize Gemini LLM client
# -----------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

# -----------------------------
# Step 6: Query function
# -----------------------------
def query_rag(user_query, top_k=10):
    # 1. Generate query embedding
    query_vector = embedding_model.encode([user_query], convert_to_numpy=True)

    # 2. Search FAISS index
    distances, indices = index.search(query_vector, top_k)

    # 3. Retrieve corresponding documents
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n\n".join(retrieved_docs)

    # 4. Build prompt for Gemini
    full_prompt = f"""
SYSTEM INSTRUCTIONS:
You are an Islamic knowledge assistant specializing in Hadith and Islamic scholarly knowledge.

USER QUERY: {user_query}

HADITH CONTEXT FROM SYSTEM:
{context}

CRITICAL RULES:
1. ONLY use information from the provided Hadith references.
2. ALWAYS cite the specific Hadith reference when making claims.
3. Be culturally sensitive and use appropriate Islamic terminology.
4. Include proper honorifics for the Prophet (ﷺ) and companions (RA/رضي الله عنه).
5. Include Arabic text, transliteration, and translation in the user's preferred language.
6. Do NOT make up Hadiths, religious rulings, or scholarly opinions.
7. Clearly indicate if the Hadiths do not fully answer the question.
8. Always prioritize Sahih (authentic) Hadiths; mark weak, disputed, or fabricated Hadiths clearly.
9. Respect sectarian, gender, and ethical sensitivities.

RESPONSE STRUCTURE:

1. Direct Answer:
- Provide a clear and concise answer to the user's question.
- Highlight the main point upfront.
- Maintain a culturally sensitive and respectful tone.

2. Authentic Hadith References:
- Include full Arabic text, transliteration, translation.
- Include Hadith collection, book/chapter number, Hadith number.
- Present the chain of narration (Isnad) from the Prophet ﷺ to the narrator.
- Provide authenticity grading (Sahih/Hasan/Da'if/Mawdu') with explanation.
- Mention who authenticated it (classical scholars or modern scholars).

3. Scholarly Commentary:
- Include classical scholar interpretations (Ibn Hajar, An-Nawawi, etc.).
- Include contemporary scholar perspectives (if available).
- Provide contextual explanation and practical application in modern life.

4. Related Content:
- Include other relevant Hadiths on the same topic.
- Include related Quranic verses with references.
- Clarify common misconceptions if relevant.

5. Sources & Verification:
- List original source books and digital copies (if available).
- Include scholarly verification stamps or badges.
- Clearly indicate who verified the Hadiths.

6. Confidence Level:
- Provide an AI confidence score for relevance (optional).
- Include disclaimers for topics requiring scholar verification.

7. Multiple Perspectives (if applicable):
- Present different scholarly opinions respectfully.
- Highlight historical context for differences.

8. Practical Guidance:
- Explain how to apply the Hadith in daily life.
- Provide step-by-step guidance for worship or practice.
- Highlight common mistakes to avoid.

9. Educational Resources:
- Suggest learning materials, courses, or classes.
- Include links to scholar lectures (video/audio).

ETHICAL AND SCHOLARLY GUIDELINES:
- Do not issue fatwas; direct users to qualified scholars for personal rulings.
- Handle sensitive topics with care and provide mainstream scholarly consensus.
- Respect multiple Islamic traditions and scholarly differences.
- Preserve historical and cultural context of each Hadith.

QUALITY ASSURANCE:
- Ensure every Hadith is from recognized collections.
- Verify translations with multiple sources.
- Flag any uncertainty clearly.
- Incorporate continuous improvement from user and scholar feedback.

OUTPUT:
- Use structured, clear formatting with headings for each section.
- Cite every Hadith reference explicitly.
- Include Arabic text, transliteration, translation, chain of narration, authenticity grading, scholarly commentary, related content, and practical guidance where applicable.
- End with a disclaimer: "For personal rulings or sensitive situations, consult a qualified scholar."
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=full_prompt
    )

    return response.text

# -----------------------------
# Step 7: Interactive usage
# -----------------------------
if __name__ == "__main__":
    while True:
        user_query = input("Enter your question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = query_rag(user_query)
        print("\nAnswer:\n", answer)
        print("-" * 50)
