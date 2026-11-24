#!/usr/bin/env python3
"""
main.py - CLI para el Classical Listening Guide Generator.
Recibe la URL de un álbum de Apple Music y genera una Guía de Escucha interactiva
en un único archivo HTML 100 % autocontenido y offline.
"""

import argparse
import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

from src.graph import create_listening_guide_graph


def setup_logging(verbose: bool = False):
    """Configura el formato y nivel de logging para la consola."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_environment():
    """Valida que existan las credenciales necesarias para los agentes de Gemini."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ ERROR: No se encontró la variable GEMINI_API_KEY (o GOOGLE_API_KEY).", file=sys.stderr)
        print("Asegúrate de definirla en tu archivo .env o en las variables de entorno del sistema:", file=sys.stderr)
        print("  GEMINI_API_KEY=tu_api_key_de_gemini\n", file=sys.stderr)
        sys.exit(1)

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        logging.info("ℹ️ TAVILY_API_KEY no detectada. Se utilizará DuckDuckGo como motor de búsqueda musicológica.")


async def run_pipeline(
    album_url: str,
    output_path: str,
    max_attempts: int = 3,
    selected_tracks: list[int] = None,
    model_name: str = "gemini-3.5-flash",
):
    """Ejecuta el pipeline completo de LangGraph."""
    logger = logging.getLogger("ClassicalListeningGuide")

    print("\n" + "=" * 65)
    print("🎼 CLASSICAL LISTENING GUIDE GENERATOR")
    print("=" * 65)
    print(f"🎵 URL del álbum: {album_url}")
    print(f"💾 Archivo destino: {output_path}")
    print(f"🤖 Modelo Gemini: {model_name}")
    print(f"🛡️ Intentos máximos por Quality Gate: {max_attempts}")
    if selected_tracks:
        print(f"🎯 Pistas seleccionadas: {selected_tracks}")
    print("-" * 65 + "\n")

    logger.info("Compilando el grafo multi-agente LangGraph...")
    graph = create_listening_guide_graph(max_attempts=max_attempts, model_name=model_name)

    initial_state = {
        "album_url": album_url,
        "metadata": None,
        "tracks": [],
        "guides": [],
        "final_html": None,
        "errors": [],
    }

    # Si se especificaron pistas seleccionadas, extraemos los metadatos y filtramos
    if selected_tracks:
        from src.extractor import extract_album_metadata
        logger.info(f"Extrayendo metadatos para filtrar pistas seleccionadas: {selected_tracks}...")
        meta = await asyncio.to_thread(extract_album_metadata, album_url)
        filtered_tracks = [t for t in meta.tracks if t.track_number in selected_tracks]
        if not filtered_tracks:
            logger.error(f"Ninguna de las pistas solicitadas {selected_tracks} existe en el álbum (total pistas: {len(meta.tracks)}).")
            sys.exit(1)
        initial_state["metadata"] = meta
        initial_state["tracks"] = filtered_tracks

    logger.info("Iniciando extracción y pipeline de investigación en paralelo...")
    final_state = await graph.ainvoke(initial_state)

    meta = final_state.get("metadata")
    guides = final_state.get("guides", [])
    errors = final_state.get("errors", [])
    final_html = final_state.get("final_html")

    if not final_html:
        logger.error("❌ El pipeline finalizó pero no se generó el HTML final.")
        if errors:
            for err in errors:
                logger.error(f"  - {err}")
        sys.exit(1)

    # 1. Determinar el nombre del compositor para la subcarpeta
    composer_clean = "Varios"
    if meta and meta.composer:
        composer_name = meta.composer.strip()
        for famous in [
            "Beethoven", "Chopin", "Mozart", "Bach", "Schubert", "Schumann",
            "Brahms", "Liszt", "Debussy", "Ravel", "Rachmaninoff", "Tchaikovsky"
        ]:
            if famous.lower() in composer_name.lower():
                composer_clean = famous
                break
        else:
            composer_clean = "".join(c for c in composer_name if c.isalnum() or c in (" ", "_", "-")).strip() or "Varios"

    base_filename = os.path.basename(output_path)
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_composer_dir = os.path.join(workspace_dir, "ListeningGuides", composer_clean)
    os.makedirs(workspace_composer_dir, exist_ok=True)
    target_workspace_path = os.path.join(workspace_composer_dir, base_filename)

    with open(target_workspace_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    # Si se pasó una ruta personalizada explícita distinta de ListeningGuides, guardarla también allí
    if os.path.abspath(output_path) != os.path.abspath(target_workspace_path):
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_html)

    # 2. Copia automática a /Users/pablovazquez/Documents/Piano/ListeningGuides/<Compositor>/
    piano_docs_dir = "/Users/pablovazquez/Documents/Piano/ListeningGuides"
    piano_target_path = None
    try:
        piano_composer_dir = os.path.join(piano_docs_dir, composer_clean)
        os.makedirs(piano_composer_dir, exist_ok=True)
        piano_target_path = os.path.join(piano_composer_dir, base_filename)
        with open(piano_target_path, "w", encoding="utf-8") as f:
            f.write(final_html)
        logger.info(f"Copia sincronizada en: {piano_target_path}")
    except Exception as e:
        logger.warning(f"No se pudo copiar automáticamente a Documents/Piano/ListeningGuides: {e}")

    print("\n" + "=" * 65)
    print("✨ ¡GUÍA DE ESCUCHA GENERADA CON ÉXITO!")
    print("=" * 65)
    if meta:
        print(f"🎶 Álbum: {meta.title}")
        print(f"👤 Compositor: {meta.composer} | Intérprete: {meta.performer}")
    print(f"📊 Pistas analizadas y aprobadas: {len(guides)}")

    warnings_count = sum(1 for g in guides if g.audit_warning)
    if warnings_count > 0:
        print(f"⚠️ Pistas compiladas con advertencia de auditoría: {warnings_count}")

    print(f"📁 Workspace: {target_workspace_path}")
    if piano_target_path:
        print(f"📁 Documentos Piano: {piano_target_path}")
    print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generador automatizado de Guías de Escucha Clásica desde Apple Music en HTML offline."
    )
    parser.add_argument(
        "url",
        help="URL completa del álbum de Apple Music (ej. https://music.apple.com/es/album/...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="guia_escucha.html",
        help="Ruta del archivo HTML generado (por defecto: guia_escucha.html)",
    )
    parser.add_argument(
        "-t",
        "--tracks",
        help="Números de pista a procesar separados por comas o rangos (ej. '1,2,3' o '1-3'). Si se omite, procesa todo el álbum.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-3.5-flash",
        help="Modelo de Gemini a utilizar (por defecto: gemini-3.5-flash)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Número máximo de intentos por pista antes de aplicar degradación elegante (por defecto: 3)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Activa logs detallados para depuración",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    validate_environment()

    selected_tracks = None
    if args.tracks:
        selected_tracks = []
        for part in args.tracks.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                selected_tracks.extend(range(int(start), int(end) + 1))
            elif part.isdigit():
                selected_tracks.append(int(part))
        selected_tracks = sorted(list(set(selected_tracks)))

    try:
        asyncio.run(
            run_pipeline(
                album_url=args.url,
                output_path=args.output,
                max_attempts=args.max_attempts,
                selected_tracks=selected_tracks,
                model_name=args.model,
            )
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Proceso cancelado por el usuario.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logging.exception(f"Error fatal durante la ejecución: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
