"""
tests/test_template_and_cli.py - Pruebas para la plantilla Jinja2 offline y el CLI main.py.
"""

import os
import subprocess
import sys
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.schemas import AlbumMetadata, MusicalMilestone, TrackGuide, TrackInfo


def test_jinja2_template_rendering():
    """Verifica que la plantilla templates/guide_template.html renderice correctamente sin CDNs."""
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("guide_template.html")

    album = AlbumMetadata(
        title="Chopin: Nocturnes",
        composer="Frédéric Chopin",
        performer="Arthur Rubinstein",
        label="RCA Red Seal",
        release_year=1965,
        album_url="https://music.apple.com/album/chopin-nocturnes",
        tracks=[
            TrackInfo(
                track_number=1,
                title="Nocturne in E-Flat Major, Op. 9 No. 2",
                duration_seconds=270,
                movement="Andante",
            )
        ],
    )

    guide = TrackGuide(
        track_number=1,
        title="Nocturne in E-Flat Major, Op. 9 No. 2",
        duration_seconds=270,
        historical_context="Compuesto entre 1830 y 1832, es uno de los nocturnos más célebres de Chopin.",
        structure_overview="Forma rondó-variación con una coda florida y expresiva.",
        milestones=[
            MusicalMilestone(
                start_second=0,
                end_second=60,
                timestamp_label="00:00 - 01:00",
                section_name="Tema Principal",
                what_to_listen="La melodía lírica y su carácter vocal de bel canto.",
                right_hand="Línea melódica ornamentada con rubato sutil.",
                left_hand="Acompañamiento en compás compuesto con acordes arpegiados regulares.",
                pedagogical_tip="Toca la mano izquierda muy regular como un lecho para la libertad de la derecha.",
            )
        ],
        source_evidence=["Chopin Complete Works Analysis."],
        audit_warning=False,
    )

    rendered = template.render(album=album, guides=[guide], total_tracks=1)

    # Verificaciones de contenido y autocontención
    assert "Chopin: Nocturnes" in rendered
    assert "Arthur Rubinstein" in rendered
    assert "Frédéric Chopin" in rendered
    assert "Mano Derecha (Melodía / Textura)" in rendered
    assert "Mano Izquierda (Bajo / Armonía)" in rendered
    assert "Consejo para el Piano" in rendered
    assert "data-start=\"0\"" in rendered
    assert "data-end=\"60\"" in rendered

    # Asegurar cero dependencias externas o CDNs
    assert "http://" not in rendered or "schema.org" in rendered  # Ningún CDN http
    assert "https://" not in rendered or "schema.org" in rendered  # Ningún CDN https (bootstrap, google fonts, cdnjs, etc)
    assert "fonts.googleapis.com" not in rendered
    assert "cdn." not in rendered


def test_main_cli_help():
    """Verifica que el comando main.py --help responda sin errores."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Guías de Escucha Clásica" in result.stdout
    assert "--output" in result.stdout
    assert "--max-attempts" in result.stdout
