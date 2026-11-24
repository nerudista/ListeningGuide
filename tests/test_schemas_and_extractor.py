"""
tests/test_schemas_and_extractor.py - Pruebas unitarias para schemas y el extractor de Apple Music.
"""

import pytest
from pydantic import ValidationError

from src.schemas import (
    AlbumMetadata,
    TrackInfo,
    MusicalMilestone,
    TrackGuide,
    AuditResult,
)
from src.extractor import parse_iso8601_duration, AppleMusicExtractor


def test_parse_iso8601_duration():
    # Minutos y segundos estándar
    assert parse_iso8601_duration("PT5M30S") == 330
    # Solo segundos
    assert parse_iso8601_duration("PT45S") == 45
    # Horas, minutos y segundos
    assert parse_iso8601_duration("PT1H2M10S") == 3730
    # Notación de dos puntos
    assert parse_iso8601_duration("05:30") == 330
    assert parse_iso8601_duration("1:05:30") == 3930
    # Cadena vacía o inválida
    assert parse_iso8601_duration("") == 0


def test_musical_milestone_validation():
    # Válido
    m = MusicalMilestone(
        start_second=0,
        end_second=45,
        timestamp_label="00:00 - 00:45",
        section_name="Introducción",
        what_to_listen="Arpegios suaves creando la atmósfera nocturna.",
        right_hand="Arpegios continuos en tresillos en registro medio.",
        left_hand="Notas fundamentales en octavas sostenidas en el bajo.",
        pedagogical_tip="Mantén el pedal sin enturbiar las notas.",
    )
    assert m.start_second == 0
    assert m.end_second == 45

    # Inválido: start_second >= end_second
    with pytest.raises(ValidationError):
        MusicalMilestone(
            start_second=50,
            end_second=50,
            timestamp_label="00:50 - 00:50",
            section_name="Error",
            what_to_listen="...",
            right_hand="...",
            left_hand="...",
        )


def test_extractor_json_ld_parsing():
    sample_html = r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Beethoven: Piano Sonatas Nos. 8, 14 & 23 - Daniel Barenboim</title>
        <script type="application/ld+json">
        {
            "@context": "http://schema.org",
            "@type": "MusicAlbum",
            "name": "Beethoven: Piano Sonatas Nos. 8, 14 & 23",
            "byArtist": [
                {"@type": "MusicGroup", "name": "Daniel Barenboim"}
            ],
            "datePublished": "1984-01-01",
            "publisher": {
                "@type": "Organization",
                "name": "Deutsche Grammophon"
            },
            "tracks": [
                {
                    "@type": "MusicRecording",
                    "name": "Piano Sonata No. 14 in C-Sharp Minor, Op. 27 No. 2 \"Moonlight\": I. Adagio sostenuto",
                    "duration": "PT5M45S",
                    "url": "https://music.apple.com/es/album/track-1"
                },
                {
                    "@type": "MusicRecording",
                    "name": "Piano Sonata No. 14 in C-Sharp Minor, Op. 27 No. 2 \"Moonlight\": II. Allegretto",
                    "duration": "PT2M15S",
                    "url": "https://music.apple.com/es/album/track-2"
                }
            ]
        }
        </script>
    </head>
    <body></body>
    </html>
    """

    extractor = AppleMusicExtractor()
    meta = extractor.extract_from_html(sample_html, source_url="https://music.apple.com/es/album/sample")

    assert meta.title == "Beethoven: Piano Sonatas Nos. 8, 14 & 23"
    assert meta.performer == "Daniel Barenboim"
    assert meta.composer == "Beethoven"
    assert meta.label == "Deutsche Grammophon"
    assert meta.release_year == 1984
    assert len(meta.tracks) == 2

    t1 = meta.tracks[0]
    assert t1.track_number == 1
    assert t1.duration_seconds == 345  # 5*60 + 45
    assert t1.work_name == 'Piano Sonata No. 14 in C-Sharp Minor, Op. 27 No. 2 "Moonlight"'
    assert t1.movement == "I. Adagio sostenuto"

    t2 = meta.tracks[1]
    assert t2.track_number == 2
    assert t2.duration_seconds == 135  # 2*60 + 15
    assert t2.movement == "II. Allegretto"


def test_track_guide_and_audit_schemas():
    guide = TrackGuide(
        track_number=1,
        title="I. Adagio sostenuto",
        duration_seconds=345,
        historical_context="Compuesta en 1801, dedicada a Giulietta Guicciardi.",
        structure_overview="Forma quasi-fantasía con tresillo de fondo constante.",
        milestones=[
            MusicalMilestone(
                start_second=0,
                end_second=60,
                timestamp_label="00:00 - 01:00",
                section_name="Introducción y Entrada de la Melodía",
                what_to_listen="El característico patrón de tresillos que evoca la luz de luna sobre el lago.",
                right_hand="Tresillos continuos sobre los que surge la melodía pausada en notas con puntillo.",
                left_hand="Bajo pedal en octavas solemnes en Do sostenido.",
            )
        ],
        source_evidence=["Tovey, D. F. A Companion to Beethoven's Pianoforte Sonatas."],
    )
    assert guide.duration_seconds == 345
    assert len(guide.milestones) == 1

    audit = AuditResult(
        approved=True,
        feedback="Tiempos coherentes sin solapamiento y evidencia suficiente.",
    )
    assert audit.approved is True
