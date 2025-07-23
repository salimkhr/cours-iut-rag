from dataloader import _add_list_item, _parse_markdown_table, CourseDataLoader

def test_headings_levels():
    md = "# Titre 1\n## Titre 2\n### Titre 3"
    loader = CourseDataLoader("fake")
    blocks = loader._parse_markdown_blocks(md)
    assert blocks == [
        {"type": "heading", "level": 1, "content": "Titre 1"},
        {"type": "heading", "level": 2, "content": "Titre 2"},
        {"type": "heading", "level": 3, "content": "Titre 3"},
    ]

def test_nested_unordered_list():
    md = "- Item 1\n  - Subitem 1.1\n    - Subsubitem 1.1.1\n- Item 2"
    loader = CourseDataLoader("fake")
    blocks = loader._parse_markdown_blocks(md)
    assert blocks == [
        {
            "type": "list",
            "items": [
                {
                    "content": "Item 1",
                    "subitems": [
                        {
                            "content": "Subitem 1.1",
                            "subitems": ["Subsubitem 1.1.1"]
                        }
                    ]
                },
                "Item 2"
            ]
        }
    ]

def test_ordered_list():
    md = "1. First\n2. Second\n  1. Nested"
    loader = CourseDataLoader("fake")
    blocks = loader._parse_markdown_blocks(md)

    assert blocks == [
        {
            "type": "ordered_list",
            "items": [
                "First",
                {
                    "content": "Second",
                    "subitems": ["Nested"]
                }
            ]
        }
    ]

def test_add_list_item_simple():
    stack = []
    _add_list_item(stack, "Item 1", indent=0)
    assert stack == ["Item 1"]

def test_add_list_item_nested():
    stack = []
    _add_list_item(stack, "Item 1", indent=0)
    _add_list_item(stack, "Subitem 1.1", indent=2)
    _add_list_item(stack, "Subsubitem 1.1.1", indent=4)
    assert stack == [
        {
            "content": "Item 1",
            "subitems": [
                {
                    "content": "Subitem 1.1",
                    "subitems": ["Subsubitem 1.1.1"]
                }
            ]
        }
    ]

def test_parse_markdown_table_with_header():
    md_lines = [
        "| Col1 | Col2 |",
        "|------|------|",
        "| A    | B    |",
        "| C    | D    |"
    ]
    result = _parse_markdown_table(md_lines)
    assert result["headers"] == ["Col1", "Col2"]
    assert result["rows"] == [["A", "B"], ["C", "D"]]

def test_parse_markdown_table_without_header():
    md_lines = [
        "| A | B |",
        "| C | D |"
    ]
    result = _parse_markdown_table(md_lines)
    assert result["headers"] == []
    assert result["rows"] == [["A", "B"], ["C", "D"]]

def test_parse_markdown_blocks_code_block():
    md = """Voici un bloc de code :

```python
print("Hello, world!")
```"""
    loader = CourseDataLoader("fake_dir")
    blocks = loader._parse_markdown_blocks(md)

    assert blocks == [
        {"type": "content", "content": "Voici un bloc de code :"},
        {'type': 'code', 'language': 'python', 'content': 'print("Hello, world!")'}
    ]

def test_parse_markdown_blocks_basic_heading_and_paragraph():
    loader = CourseDataLoader("fake_dir")
    md = "# Titre\n\nCeci est un paragraphe."
    blocks = loader._parse_markdown_blocks(md)
    assert blocks == [
        {"type": "heading", "level": 1, "content": "Titre"},
        {"type": "content", "content": "Ceci est un paragraphe."}
    ]