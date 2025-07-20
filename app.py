from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import ollama
import threading
import time
from datetime import datetime
from rag_system import VectorRAG

app = Flask(__name__)
CORS(app)

# Configuration
CORPUS_DIR = "./rag/corpus"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "codellama:7b"

# Instance globale du système RAG
rag_system = None
rag_lock = threading.Lock()


def initialize_rag():
    """Initialise le système RAG de manière thread-safe"""
    global rag_system
    with rag_lock:
        if rag_system is None:
            print("🚀 Initialisation du système RAG...")
            rag_system = VectorRAG(model_name=MODEL_NAME, corpus_dir=CORPUS_DIR)


# Thread de surveillance des modifications
def watch_corpus_changes():
    """Thread qui surveille les changements dans le corpus"""
    while True:
        try:
            if rag_system:
                rag_system._check_and_reload_if_needed()
            time.sleep(10)  # Vérification toutes les 10 secondes
        except Exception as e:
            print(f"❌ Erreur dans la surveillance du corpus: {e}")
            time.sleep(30)  # Attendre plus longtemps en cas d'erreur


# Démarrage du thread de surveillance
watcher_thread = threading.Thread(target=watch_corpus_changes, daemon=True)
watcher_thread.start()


# @app.before_first_request
# def startup():
#     """Initialisation lors du premier accès"""
#     initialize_rag()


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé avec statistiques détaillées"""
    try:
        if rag_system is None:
            initialize_rag()

        stats = rag_system.get_stats()

        # Test de connectivité Ollama
        ollama_status = "unknown"
        try:
            ollama.list()
            ollama_status = "connected"
        except Exception:
            ollama_status = "disconnected"

        return jsonify({
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "rag_system": stats,
            "ollama_status": ollama_status,
            "ollama_model": OLLAMA_MODEL
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/corpus/info', methods=['GET'])
def corpus_info():
    """Informations détaillées sur le corpus"""
    try:
        if rag_system is None:
            initialize_rag()

        files_info = rag_system.data_loader.list_corpus_files()
        stats = rag_system.get_stats()

        return jsonify({
            "corpus_directory": CORPUS_DIR,
            "files": files_info,
            "total_files": len(files_info),
            "system_stats": stats
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/corpus/reload', methods=['POST'])
def reload_corpus():
    """Force le rechargement du corpus"""
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


@app.route('/search', methods=['POST'])
def search_only():
    """Endpoint pour la recherche seule (sans LLM)"""
    try:
        if rag_system is None:
            initialize_rag()

        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 5)

        if not query:
            return jsonify({"error": "Query vide"}), 400

        with rag_lock:
            results = rag_system.search_similar(query, k=k)

        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat sans streaming"""
    try:
        if rag_system is None:
            initialize_rag()

        data = request.get_json()
        message = data.get('message', '').strip()
        k = data.get('k', 3)  # Nombre de documents à récupérer

        if not message:
            return jsonify({"error": "Message vide"}), 400

        # Recherche RAG avec thread safety
        with rag_lock:
            results = rag_system.search_similar(message, k=k)
            prompt = rag_system.build_rag_prompt(results, message)

        # Appel Ollama
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            return jsonify({
                "response": response["message"]["content"],
                "sources": [{
                    "file": r["file"],
                    "type": r["type"],
                    "similarity_score": r["similarity_score"],
                    "content_preview": r["content"][:150] + "..." if len(r["content"]) > 150 else r["content"]
                } for r in results],
                "query": message,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({"error": f"Erreur Ollama: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Chat avec streaming"""
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
            results = rag_system.search_similar(message, k=k)
            prompt = rag_system.build_rag_prompt(results, message)

        def generate():
            try:
                # Stream depuis Ollama
                stream = ollama.chat(
                    model="codellama:7b",
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