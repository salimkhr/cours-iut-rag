import pytest

from text_processor import TextProcessor


@pytest.fixture
def processor():
    return TextProcessor(chunk_size=50, min_chunk_size=6, overlap=5)

def test_clean_text(processor):
    dirty_text = "Hello\x00 world!   \n\nThis is a \x07test."
    cleaned = processor._clean_text(dirty_text)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "  " not in cleaned
    assert cleaned == "Hello world! This is a test."

def test_split_by_paragraphs(processor):
    text = "Paragraphe un.\n\nParagraphe deux. Phrase deux."
    paragraphs = processor._split_by_paragraphs(text)
    assert len(paragraphs) == 2
    assert paragraphs[0] == "Paragraphe un."
    assert paragraphs[1] == "Paragraphe deux. Phrase deux."

def test_split_by_sentences(processor):
    text = "Phrase une. Phrase deux! Phrase trois?"
    chunks = processor._split_by_sentences(text)
    # Chaque chunk <= chunk_size, donc peut être regroupé
    # Ici chunk_size=50, texte assez court => 1 chunk attendu
    assert len(chunks) == 1
    assert "Phrase une." in chunks[0]

def test_force_split(processor):
    long_text = " ".join(["mot"] * 20)  # 20 mots "mot"
    chunks = processor._force_split(long_text)
    # Chaque chunk <= chunk_size=50, donc plusieurs chunks attendus
    assert all(len(c) <= processor.chunk_size for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 20

def test_post_process_chunks(processor):
    chunks = ["chunk1", "chunk2", "chunk1 ", "short"]
    # min_chunk_size=6 donc "short" sera filtré
    processed = processor._post_process_chunks(chunks)
    # "chunk1" (appearing twice) doit être dédupliqué
    print(len(processed))
    assert "chunk1" in processed
    assert "chunk2" in processed
    assert "short" not in processed
    assert len(processed) == 2


def test_chunk_text_short_paragraph(processor):
    text = "Petit paragraphe court."
    chunks = processor.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Petit paragraphe court."

def test_chunk_text_long_paragraph(processor):
    text = " ".join(["Phrase une."] * 10)  # Long texte à découper en phrases
    chunks = processor.chunk_text(text)
    assert all(len(c) <= processor.chunk_size for c in chunks)

def test_prepare_texts_for_embedding(processor):
    data = [
        {
            "title": "Cours 1",
            "file": "cours1.md",
            "blocks": [
                {"type": "text", "content": "Phrase une. Phrase deux."},
                {"type": "code", "content": "print('hello')"},
            ]
        }
    ]
    texts, meta = processor.prepare_texts_for_embedding(data)
    assert len(texts) == len(meta)
    assert all("file" in m and "course_title" in m and "content" in m for m in meta)

def test_get_chunk_stats(processor):
    chunks = ["abcde", "fghij", "klmno"]
    stats = processor.get_chunk_stats(chunks)
    assert stats["count"] == 3
    assert stats["min_length"] == 5
    assert stats["max_length"] == 5
    assert stats["total_chars"] == 15
    assert abs(stats["avg_length"] - 5.0) < 1e-6

def test_get_chunk_stats_empty(processor):
    stats = processor.get_chunk_stats([])
    assert stats["count"] == 0