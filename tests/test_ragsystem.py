import os
import shutil
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock

from src.rag_system import VectorRAG

@pytest.fixture
def fake_corpus_dir():
    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(temp_dir, "test.md"), "w", encoding="utf-8") as f:
        f.write("# Titre\n\nContenu test de cours.")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_rag(fake_corpus_dir):
    rag = VectorRAG("paraphrase-MiniLM-L6-v2", fake_corpus_dir)

    # Mock le modèle pour ne pas vraiment charger SentenceTransformer
    rag.model = MagicMock()
    rag.model.encode.side_effect = lambda texts, **kwargs: np.random.rand(len(texts), 384)

    return rag

def test_search_returns_results(mock_rag):
    query = "cours test"
    results = mock_rag.search_similar(query, k=3)

    assert isinstance(results, list)
    assert all("content" in r for r in results)

def test_manual_cosine_similarity(mock_rag):
    mock_rag.model.encode.side_effect = lambda texts, **kwargs: np.array([[1, 0, 0], [1, 0, 0]])
    score = mock_rag.calculate_cosine_similarity_manual("a", "b")
    assert score == pytest.approx(1.0)

def test_prompt_generation(mock_rag):
    fake_results = [
        {"content": "Voici un test de contenu", "file": "chapitre1.md", "type": "paragraph", "cosine_similarity": 0.85}
    ]
    question = "Qu'est-ce que le contenu test ?"
    prompt = mock_rag.build_rag_prompt(fake_results, question)

    assert "### Extrait 1" in prompt
    assert "Voici un test de contenu" in prompt
    assert question in prompt

def test_force_reload(mock_rag):
    prev_timestamp = mock_rag.last_modified
    mock_rag.force_reload()
    assert mock_rag.last_modified >= prev_timestamp