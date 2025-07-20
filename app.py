import os
import json
import ollama

# 1. Charger tous les fichiers JSON dans rag/corpus
def load_all_course_blocks(directory="./rag/corpus"):
    course_blocks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                with open(full_path, encoding="utf-8") as f:
                    data = json.load(f)
                    data["file"] = os.path.relpath(full_path, directory)
                    course_blocks.append(data)
    return course_blocks

# 2. Rechercher les blocs de texte pertinents (recherche simple par mot-clé)
def search_courses(question, all_courses):
    matches = []
    keywords = question.lower().split()
    for course in all_courses:
        for block in course["blocks"]:
            text = block.get("content", "").lower()
            if any(word in text for word in keywords):
                matches.append((course["file"], block))
    return matches

# 3. Construire un prompt clair à partir des blocs trouvés
def build_rag_prompt(matches, question):
    context = ""
    for filename, block in matches:
        context += f"\n### {filename} – {block['type']}\n{block['content']}\n"
    prompt = f"""
Voici des extraits de cours universitaires :

{context}

Réponds de manière claire et structurée à la question suivante, uniquement en te basant sur ces extraits :

→ {question}
"""
    return prompt.strip()

# 4. Envoyer le prompt à Ollama
def ask_ollama(prompt, model="codellama:7b"):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response["message"]["content"]

# 5. App principale
if __name__ == "__main__":
    question = input("💬 Pose ta question sur le cours :\n> ")
    print("\n🔍 Recherche dans les fichiers de cours...\n")

    courses = load_all_course_blocks()
    matches = search_courses(question, courses)

    if not matches:
        print("❌ Aucun extrait de cours pertinent trouvé.")
    else:
        prompt = build_rag_prompt(matches, question)
        print("📡 Envoi à Ollama...\n")
        response = ask_ollama(prompt)
        print("✅ Réponse :\n")
        print(response)
