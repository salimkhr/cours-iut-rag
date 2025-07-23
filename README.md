# **Moteur RAG (Retrieval-Augmented Generation)**
Ce projet implémente un **moteur RAG** en Python, permettant de répondre à des questions en s'appuyant sur un corpus de documents vectorisés à partir de fichiers Markdown.
## 📂 **Structure du projet**
### Principaux fichiers :
- **`app.py` **: Script principal contenant l’API Flask pour exposer les services RAG, comme la recherche et le chat basé sur un LLM.
- **`rag_system.py` **: Moteur RAG basé sur des embeddings vectoriels et FAISS pour les recherches efficaces.
- **`text_processor.py` **: Découpage, nettoyage et optimisation des blocs de texte pour l’indexation.
- **`dataloader.py` **: Chargement et traitement des fichiers Markdown.
- **`rag/markdown/`** : Répertoire contenant les fichiers Markdown (corpus d'entraînement, par exemple des cours).
- **`venv/`** : Environnement virtuel Python (non versionné).
- **`rag/index/`** _(optionnel)_ : Dossier pour stocker les index FAISS persistés.

## 🛠️ **Fonctionnalités principales**
✔ Chargement des fichiers de cours en format Markdown
✔ Nettoyage du texte et découpage en chunks exploitables pour l'embedding
✔ Embedding avec SentenceTransformers (`all-MiniLM-L6-v2` par défaut)
✔ Recherche vectorielle performante grâce à FAISS
✔ Génération d’un prompt enrichi pour interagir avec un LLM (Ollama, OpenAI, etc.)
✔ Reload automatique du corpus lorsqu’une modification est détectée
✔ Points d’extension pour intégrer une interface externe (React/Next.js, etc.)
## ⚙️ **Lancer le projet**
1. Créez un environnement virtuel (si ce n’est pas déjà fait) :
``` bash
   python -m venv venv
```
1. Activez-le :
    - Sous **Windows** :
``` bash
     venv\Scripts\activate
```
- Sous **Linux/macOS** :
``` bash
     source venv/bin/activate
```
1. Installez les dépendances requises :
``` bash
   pip install -r requirements.txt
```
1. Placez vos fichiers Markdown dans le dossier suivant :
``` 
   rag/markdown/
```
Format attendu (Markdown) :
``` markdown
   # Titre du cours
   ## Sous-section 1
   Contenu de cours en paragraphes...

   - Exemple de liste
   - Une autre entrée

   ## Sous-section 2
   Tableau :
   | Colonne A | Colonne B |
   |-----------|-----------|
   | Ligne 1   | Valeur 1  |
```
Ces fichiers seront transformés en blocs structurés, prêts pour l'embedding.
1. Lancez l’application :
``` bash
   python app.py
```
1. Faites une requête à l’API (par exemple via Postman ou un client HTTP). Exemple pour rechercher une question :
``` json
   POST /chat
   {
     "message": "Explique le modèle de boîte en CSS",
     "k": 5
   }
```
Le système retournera les réponses et les extraits similaires provenant du corpus vectorisé.
## 🔗 **Connexion avec une interface Next.js**
Ce backend expose une API compatible avec toute interface client.
Par exemple : une **application React/Next.js** peut envoyer la question à l'API et afficher la réponse du système augmentée d'extraits du corpus.
## 📦 **Dépendances principales**
- `Flask` (serveur web/API)
- `sentence-transformers` (modèles d’embeddings)
- `faiss-cpu` (recherche vectorielle en haute performance)
- `numpy` (calcul matriciel)
- `pandas` _(optionnel, pour explorer le corpus)_

## 🧠 **Fonctionnement résumé**
1. **Lecture des fichiers Markdown** : Les cours ou documents sont chargés depuis le dossier spécifié.
2. **Nettoyage et découpage en chunks** : Texte découpé pour une meilleure efficacité des embeddings.
3. **Embedding des chunks** : Les chunks sont transformés en vecteurs numériques grâce à un modèle SentenceTransformer.
4. **Indexation FAISS** : Les vecteurs sont insérés dans un index FAISS pour des recherches rapides.
5. **Recherche vectorielle** : Questions ou requêtes sont transformées en vecteurs, puis FAISS identifie les blocs correspondants.
6. **Génération d’un prompt RAG** : Compile les extraits trouvés pour créer un contexte structuré.
