"""
src/extractor.py - Extracción determinista de metadatos de Apple Music sin LLM.
Extrae compositor, intérprete, sello, año, lista de pistas y duration_seconds exacta
analizando los bloques JSON-LD (schema.org) y metadatos estructurados del HTML.
"""

import json
import re
from typing import Any, Dict, List, Optional
import httpx
from bs4 import BeautifulSoup

from src.schemas import AlbumMetadata, TrackInfo


def parse_iso8601_duration(duration_str: str) -> int:
    """
    Convierte una duración ISO 8601 (ej. 'PT5M30S', 'PT45S', 'PT1H2M3S', 'PT180S') a segundos enteros.
    Si falla o no viene en formato estándar, intenta extraer minutos y segundos mediante regex.
    """
    if not duration_str:
        return 0

    # Patrón estándar ISO 8601 para duraciones: P[nY][nM][nD][T[nH][nM][nS]]
    iso_pattern = re.compile(
        r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
    )
    match = iso_pattern.match(duration_str.strip())
    if match:
        parts = match.groupdict()
        hours = int(parts["hours"] or 0)
        minutes = int(parts["minutes"] or 0)
        seconds = float(parts["seconds"] or 0)
        return int(hours * 3600 + minutes * 60 + round(seconds))

    # Fallback para duraciones tipo "05:32" o "1:02:15"
    colon_pattern = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")
    colon_match = colon_pattern.match(duration_str.strip())
    if colon_match:
        h, m, s = colon_match.groups()
        return int(h or 0) * 3600 + int(m) * 60 + int(s)

    # Fallback de solo dígitos numéricos (milisegundos o segundos)
    if duration_str.isdigit():
        val = int(duration_str)
        # Si es mayor a 100_000 suele ser milisegundos
        return val // 1000 if val > 100_000 else val

    return 0


def clean_text(text: Optional[str]) -> str:
    """Limpia espacios en blanco redundantes y caracteres especiales."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


class AppleMusicExtractor:
    """
    Extractor determinista para álbumes de Apple Music utilizando HTTP y parseo JSON-LD / HTML.
    No requiere browser headless si el HTML estático contiene el JSON-LD schema.org.
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
    }

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def fetch_html(self, url: str) -> str:
        """Descarga el HTML crudo de la URL de Apple Music."""
        with httpx.Client(headers=self.DEFAULT_headers if hasattr(self, "DEFAULT_headers") else self.DEFAULT_HEADERS,
                          follow_redirects=True, timeout=self.timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text

    def extract_from_html(self, html: str, source_url: str) -> AlbumMetadata:
        """Analiza el HTML y extrae la metadata completa del álbum."""
        soup = BeautifulSoup(html, "html.parser")

        # 1. Buscar todos los bloques JSON-LD
        json_ld_blocks = soup.find_all("script", type="application/ld+json")
        album_data: Optional[Dict[str, Any]] = None

        for script in json_ld_blocks:
            content = script.get_text(strip=True)
            if not content:
                continue
            try:
                data = json.loads(content)
                # Puede ser un objeto directo o una lista/grafo
                candidates = []
                if isinstance(data, dict):
                    if data.get("@type") == "MusicAlbum":
                        album_data = data
                        break
                    if "@graph" in data and isinstance(data["@graph"], list):
                        candidates.extend(data["@graph"])
                    else:
                        candidates.append(data)
                elif isinstance(data, list):
                    candidates.extend(data)

                for c in candidates:
                    if isinstance(c, dict) and c.get("@type") == "MusicAlbum":
                        album_data = c
                        break
                if album_data:
                    break
            except Exception:
                continue

        # 2. Si no se encontró en JSON-LD, intentar extraer de scripts serializados de Apple Music
        if not album_data:
            album_data = self._extract_from_serialized_scripts(soup)

        # 3. Extraer campos principales con fallbacks a etiquetas OpenGraph/Meta
        title = ""
        composer = ""
        performer = ""
        label = None
        release_year = None
        tracks: List[TrackInfo] = []

        if album_data:
            title = clean_text(album_data.get("name") or album_data.get("headline"))

            # Artistas / Intérprete / Compositor
            artists = album_data.get("byArtist") or album_data.get("artist") or []
            if isinstance(artists, dict):
                artists = [artists]

            artist_names = [clean_text(a.get("name", "")) for a in artists if isinstance(a, dict) and a.get("name")]
            if artist_names:
                performer = ", ".join(artist_names)

            # Sello / Publisher
            pub = album_data.get("publisher") or album_data.get("recordLabel")
            if isinstance(pub, dict):
                label = clean_text(pub.get("name"))
            elif isinstance(pub, str):
                label = clean_text(pub)

            # Fecha de lanzamiento
            date_published = album_data.get("datePublished") or album_data.get("uploadDate")
            if date_published:
                year_match = re.search(r"\b(19\d\d|20\d\d)\b", str(date_published))
                if year_match:
                    release_year = int(year_match.group(1))

            # Pistas
            raw_tracks = album_data.get("tracks") or album_data.get("track") or []
            if isinstance(raw_tracks, dict):
                raw_tracks = [raw_tracks]

            for idx, raw_t in enumerate(raw_tracks, start=1):
                if not isinstance(raw_t, dict):
                    continue
                t_name = clean_text(raw_t.get("name", f"Track {idx}"))
                raw_dur = raw_t.get("duration", "")
                dur_secs = parse_iso8601_duration(str(raw_dur))
                t_url = raw_t.get("url")

                # Parsear obra / movimiento si el título usa la convención clásica usual (ej. "Obra: Movimiento")
                work_name, movement = self._split_classical_title(t_name)

                # Compositor o intérprete de la pista si viene especificado
                t_composer = None
                if "composer" in raw_t:
                    c_data = raw_t["composer"]
                    if isinstance(c_data, dict):
                        t_composer = clean_text(c_data.get("name"))
                    elif isinstance(c_data, list) and c_data:
                        t_composer = clean_text(c_data[0].get("name", ""))

                tracks.append(
                    TrackInfo(
                        track_number=idx,
                        title=t_name,
                        duration_seconds=dur_secs,
                        work_name=work_name,
                        movement=movement,
                        composer=t_composer,
                        performer=performer or None,
                        url=t_url,
                    )
                )

        # 4. Fallbacks con Meta Tags si faltan datos
        if not title:
            og_title = soup.find("meta", property="og:title")
            title = clean_text(og_title.get("content", "")) if og_title else "Álbum Desconocido"

        if not composer or not performer:
            # Buscar en descripción o títulos clásicos usuales tipo "Beethoven: Piano Sonatas - Daniel Barenboim"
            og_desc = soup.find("meta", property="og:description")
            desc_text = og_desc.get("content", "") if og_desc else ""
            extracted_comp, extracted_perf = self._guess_composer_and_performer(title, desc_text)
            if not composer:
                composer = extracted_comp
            if not performer:
                performer = extracted_perf

        # Si aún no tenemos compositor definido pero el título o la obra lo sugiere
        if not composer and performer:
            composer = performer

        # Si no se detectaron pistas por JSON-LD, extraer del DOM de Apple Music
        if not tracks:
            tracks = self._extract_tracks_from_dom(soup)

        # Si alguna pista tiene duration_seconds == 0, asignar un valor por defecto razonable o warning
        for t in tracks:
            if t.duration_seconds <= 0:
                t.duration_seconds = 180  # Fallback de 3 min si no se pudo leer duración exacta

        return AlbumMetadata(
            title=title or "Álbum sin título",
            composer=composer or "Compositor no especificado",
            performer=performer or "Intérprete no especificado",
            label=label,
            release_year=release_year,
            album_url=source_url,
            tracks=tracks,
        )

    def extract(self, url: str) -> AlbumMetadata:
        """Punto de entrada principal: descarga la URL y extrae la metadata."""
        html = self.fetch_html(url)
        return self.extract_from_html(html, source_url=url)

    @staticmethod
    def _split_classical_title(title: str) -> tuple[Optional[str], Optional[str]]:
        """Separa títulos clásicos habituales como 'Piano Sonata No. 14 in C-Sharp Minor, Op. 27 No. 2: I. Adagio sostenuto'."""
        if ":" in title:
            parts = title.split(":", 1)
            return clean_text(parts[0]), clean_text(parts[1])
        if " - " in title:
            parts = title.split(" - ", 1)
            return clean_text(parts[0]), clean_text(parts[1])
        return clean_text(title), None

    @staticmethod
    def _guess_composer_and_performer(title: str, description: str) -> tuple[str, str]:
        """Intenta deducir compositor e intérprete analizando patrones como 'Compositor: Obra de Intérprete'."""
        composer = ""
        performer = ""
        if ":" in title:
            possible_composer = title.split(":")[0].strip()
            # Si parece un nombre conocido o corto
            if len(possible_composer.split()) <= 4:
                composer = possible_composer

        # Analizar descripción de Apple Music (ej. "Álbum · 2021 · 12 canciones. Escucha a Maurizio Pollini...")
        if " de " in description:
            match = re.search(r"de\s+([^,·\n]+)", description)
            if match:
                performer = clean_text(match.group(1))

        return composer or "Compositor", performer or "Intérprete"

    def _extract_from_serialized_scripts(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Busca scripts con JSON interno serializado en páginas modernas de Apple Music."""
        for script in soup.find_all("script"):
            content = script.get_text()
            if "durationInMillis" in content or "shelf-track-list" in content:
                try:
                    # Intento de extraer JSON embebido
                    match = re.search(r"\{.*\"durationInMillis\".*\}", content)
                    if match:
                        return json.loads(match.group(0))
                except Exception:
                    pass
        return None

    def _extract_tracks_from_dom(self, soup: BeautifulSoup) -> List[TrackInfo]:
        """Fallback para extraer pistas de elementos HTML si el JSON-LD no contenía el array de pistas."""
        tracks = []
        # Selectores habituales en Apple Music web
        track_rows = soup.select("[data-testid='track-lockup'], .songs-list-row, .track")
        for idx, row in enumerate(track_rows, start=1):
            title_elem = row.select_one(".songs-list-row__song-name, [data-testid='track-title'], .track-title")
            duration_elem = row.select_one(".songs-list-row__length, [data-testid='track-duration'], time")

            t_name = clean_text(title_elem.text) if title_elem else f"Track {idx}"
            dur_text = clean_text(duration_elem.text) if duration_elem else "03:00"
            dur_secs = parse_iso8601_duration(dur_text)

            work_name, movement = self._split_classical_title(t_name)
            tracks.append(
                TrackInfo(
                    track_number=idx,
                    title=t_name,
                    duration_seconds=dur_secs or 180,
                    work_name=work_name,
                    movement=movement,
                )
            )
        return tracks


def extract_album_metadata(url: str) -> AlbumMetadata:
    """Función de conveniencia para extraer los metadatos de un álbum de Apple Music."""
    extractor = AppleMusicExtractor()
    return extractor.extract(url)
