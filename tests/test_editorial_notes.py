import json

from bs4 import BeautifulSoup

from src.extractor import AppleMusicExtractor


def _page_with_serialized_data(data: dict) -> str:
    return (
        "<html><head></head><body>"
        f'<script type="application/json" id="serialized-server-data">{json.dumps(data)}</script>'
        "</body></html>"
    )


def test_extracts_editorial_paragraph_from_serialized_data():
    data = {
        "data": [
            {
                "data": {
                    "sections": [
                        {
                            "items": [
                                {
                                    "modalPresentationDescriptor": {
                                        "headerTitle": "Sobre este álbum",
                                        "paragraphText": "Primer párrafo.\n\nSegundo párrafo.",
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        ]
    }
    soup = BeautifulSoup(_page_with_serialized_data(data), "html.parser")
    assert AppleMusicExtractor._extract_editorial_notes(soup) == "Primer párrafo.\n\nSegundo párrafo."


def test_returns_none_when_no_editorial_block():
    soup = BeautifulSoup(_page_with_serialized_data({"data": []}), "html.parser")
    assert AppleMusicExtractor._extract_editorial_notes(soup) is None


def test_returns_none_without_serialized_script():
    soup = BeautifulSoup("<html><body></body></html>", "html.parser")
    assert AppleMusicExtractor._extract_editorial_notes(soup) is None
