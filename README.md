Ce projet met en œuvre un moteur RAG (Retrieval-Augmented Generation) en Python, permettant de répondre à des questions en s'appuyant sur des documents de cours vectorisés (HTML, CSS, React, etc.).

📂 Structure du projet :
------------------------

- `app.py`               : Script principal pour lancer une requête ou interagir avec le système
- `rag_system.py`        : Moteur RAG basé sur embeddings + FAISS
- `text_processor.py`    : Découpage et nettoyage des blocs de texte
- `dataloader.py`        : Chargement des fichiers `.json` contenant les contenus de cours
- `rag/corpus/`          : Contient les fichiers de cours (format JSON)
- `venv/`                : Environnement virtuel Python (non versionné)
- `rag/index/`           : (optionnel) dossier pour stocker un index persisté

🛠️ Fonctionnalités principales :
-------------------------------

✔ Chargement automatique des fichiers de cours JSON  
✔ Nettoyage, découpage en chunks, déduplication  
✔ Embedding avec SentenceTransformers (`all-MiniLM-L6-v2`)  
✔ Recherche vectorielle rapide avec FAISS  
✔ Construction d’un prompt clair pour l’LLM  
✔ Reload automatique en cas de modification du corpus  
✔ Compatible avec une interface externe (ex: Next.js)

⚙️ Lancer le projet :
---------------------

1. Crée ton environnement virtuel (si ce n’est pas déjà fait) : `python -m venv venv`
2. Active-le : sous Windows : `venv\Scripts\activate`, sous linux `source venv/bin/activate`
3. Installe les dépendances : `pip install -r requirements.txt`
4. Place tes fichiers `.json` dans `rag/corpus/`.  
   Format attendu :
```json
{
  "title": "HTML & CSS",
  "file": "RappelHtml.tsx",
  "blocks": [
    { "type": "cours", "content": "Voici un rappel sur les balises HTML..." },
    ...
  ]
}
```
5. Lance l’application : `python app.py`
6. Entrez votre question : Comment fonctionne le modèle de boîte en CSS ?
Le moteur cherchera les extraits les plus pertinents dans le corpus vectorisé, construira un prompt, et pourra l’envoyer à un LLM (Ollama, OpenAI, etc.).

🔗 Connexion avec une interface Next.js :
-------------------------------
Ce backend peut être utilisé via API (FastAPI, Flask...) pour alimenter une interface React/Next.js.

Le frontend envoie la question → le backend retourne la réponse générée et les sources citées.

📦 Dépendances principales :
-------------------------------
sentence-transformers

faiss-cpu

numpy

tqdm

🧠 Fonctionnement résumé :
-------------------------------
Lecture des fichiers JSON (cours)

Nettoyage & découpage en chunks

Embedding des chunks

Indexation FAISS

Recherche vectorielle sur question

Construction du prompt

Utilisation du LLM pour générer une réponse

🔄 Reload intelligent :
-------------------------------
Le système détecte automatiquement les changements dans rag/corpus et reconstruit l’index au besoin. Un rechargement forcé est aussi possible via : `rag.force_reload()`
