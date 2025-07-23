import os
import time

from flask import Flask, request, jsonify, Response, abort
from flask_cors import CORS
import json
import ollama
import threading
from datetime import datetime
from rag_system import VectorRAG
from dotenv import load_dotenv

load_dotenv(".env.local")

app = Flask(__name__)
CORS(app)

# Configuration
CORPUS_DIR = os.getenv("CORPUS_DIR")
MODEL_NAME = os.getenv("MODEL_NAME")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
PYTHON_API_KEY = os.getenv("PYTHON_API_KEY")

# Instance globale du système RAG
rag_system = None
rag_lock = threading.Lock()

def initialize_rag():
    """
    Initialise le système global RAG (Retrieval-Augmented Generation) de manière sécurisée
    en multithreading grâce à un verrou. Cette fonction garantit que le système n’est
    initialisé qu’une seule fois afin d’éviter des opérations redondantes.

    :global rag_system: Variable globale représentant le système RAG. Initialisée si elle ne l’est pas déjà.
    :global rag_lock: Objet verrou utilisé pour assurer une initialisation thread-safe.

    :return: None
    """
    global rag_system
    with rag_lock:
        if rag_system is None:
            print("🚀 Initialisation du système RAG...")
            rag_system = VectorRAG(model_name=MODEL_NAME, corpus_dir=CORPUS_DIR)

@app.before_request
def check_api_key():
    """
    Cette fonction est un gestionnaire Flask exécuté avant chaque requête,
    qui vérifie la clé API présente dans les en-têtes de la requête.
    Elle renforce la sécurité en s’assurant que la plupart des appels aux endpoints
    contiennent une clé valide. Les requêtes vers certaines routes exemptées
    passent outre cette validation.

    :param exempt_routes: Liste des routes exclues de la validation de la clé API.
        Par exemple, la route "/docs" est actuellement exemptée.
    :type exempt_routes: list

    :return: None
    """
    exempt_routes = ["/docs"]
    if request.path in exempt_routes:
        return  # ne vérifie pas
    api_key = request.headers.get("x-api-key")

    expected = PYTHON_API_KEY
    if not api_key or api_key != expected:
        abort(403)

@app.route('/corpus/reload', methods=['POST'])
def reload_corpus():
    """
    Gère le rechargement du corpus en réinitialisant le système RAG et en forçant une recharge,
    tout en garantissant la sécurité des threads grâce à des verrous.

    :raise Exception: Capture et gère toute erreur inattendue survenant pendant le rechargement.

    :return: Une réponse JSON contenant un message de succès, les statistiques actualisées du corpus,
         et un horodatage de l’opération de rechargement. En cas d’erreur, une réponse JSON
         avec un message d’erreur et un code HTTP 500 est renvoyée.
    :rtype: flask.Response
    """
    try:
        if rag_system is None:
            initialize_rag()

        with rag_lock:
            rag_system.force_reload()

        stats = rag_system.get_stats()

        return jsonify({
            "message": "Corpus rechargé avec succès",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """
    Gère les événements envoyés par le serveur (SSE  Server-Sent Events) pour un endpoint de chat,
    en utilisant un système RAG (Retrieval-Augmented Generation) pour répondre aux requêtes des utilisateurs.
    La fonction s’assure que le système RAG est initialisé, récupère la saisie utilisateur,
    effectue une recherche basée sur la similarité via RAG, construit une invite (prompt)
    et diffuse le contenu de la réponse en continu, incluant les sources pertinentes.

    :raises: Exception
        Si l’initialisation du système RAG échoue ou si une erreur survient pendant l’exécution.

    :parameters:
        Aucun

    :return: Objet réponse Flask qui stream les données pour le SSE, avec le contenu incrémental
             et les informations sur les sources.
    :rtype: Response
    """
    try:
        if rag_system is None:
            initialize_rag()

        data = request.get_json()
        message = data.get('message', '').strip()
        k = data.get('k', 3)

        if not message:
            error_json = json.dumps({"error": "Message vide"})
            return Response(
                f"data: {error_json}\n\n",
                mimetype='text/event-stream'  # event-stream est mieux pour SSE
            )

        # Recherche RAG
        with rag_lock:
            results = rag_system.search_similar(message, k=k, include_neighbors=True)
            prompt = rag_system.build_rag_prompt(results, message)

        def generate():
            try:
                # Stream depuis Ollama
                stream = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )

                for chunk in stream:
                    if 'message' in chunk:
                        content = chunk['message'].get('content', '')
                        if content:
                            data_json = json.dumps({"content": content, "done":0})
                            yield f"data: {data_json}\n\n"

                # Envoi des sources à la fin
                sources_data = {
                    "sources": [
                        {
                            "file": r["file"],
                            "type": r["type"],
                            "score": r["similarity_score"],
                            "done":1
                        }
                        for r in results
                    ]
                }
                yield f"data: {json.dumps(sources_data)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                error_json = json.dumps({"error": str(e)})
                yield f"data: {error_json}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache'}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)