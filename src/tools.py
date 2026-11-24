"""
src/tools.py - Herramientas de búsqueda musicológica y resiliencia para llamadas a Gemini.
Implementa:
1. Búsqueda con Tavily con fallback automático a DuckDuckGo.
2. Wrapper asíncrono no bloqueante.
3. Resiliencia y reintentos exponenciales para Rate Limits de Gemini (429 / ResourceExhausted) con tenacity.
4. Validación determinista matemática de hitos y solapamientos.
"""

import asyncio
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

load_dotenv()
logger = logging.getLogger(__name__)


def is_rate_limit_or_transient_error(exception: BaseException) -> bool:
    """Detecta si la excepción corresponde a un rate limit (429, ResourceExhausted) o error transitorio de red."""
    msg = str(exception).lower()
    transient_indicators = [
        "429",
        "resourceexhausted",
        "resource_exhausted",
        "rate limit",
        "ratelimit",
        "quota exceeded",
        "too many requests",
        "temporarily unavailable",
        "503",
        "service unavailable",
        "disconnected",
        "server disconnected",
        "remote protocol",
        "remoteprotocolerror",
        "connection reset",
        "connection error",
        "connecterror",
        "timeout",
        "timed out",
        "readtimeout",
    ]
    return any(indicator in msg for indicator in transient_indicators)


def search_musicology(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Busca información musicológica real.
    Intenta usar Tavily si TAVILY_API_KEY está configurada, o recurre a DuckDuckGo como fallback.
    Devuelve lista de diccionarios con keys: 'title', 'snippet', 'url'.
    """
    results: List[Dict[str, str]] = []
    tavily_key = os.getenv("TAVILY_API_KEY")

    # 1. Intentar con Tavily si hay API Key disponible
    if tavily_key and tavily_key.strip():
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key.strip())
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            )
            raw_items = response.get("results", [])
            for item in raw_items:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("content", ""),
                        "url": item.get("url", ""),
                    }
                )
            if results:
                logger.info(f"Búsqueda Tavily exitosa para: '{query}' ({len(results)} resultados)")
                return results
        except Exception as e:
            logger.warning(f"Error o cuota agotada en Tavily ({e}). Activando fallback a DuckDuckGo...")

    # 2. Fallback a DuckDuckGo
    try:
        from duckduckgo_search import DDGS

        ddgs = DDGS()
        ddg_results = ddgs.text(query, max_results=max_results)
        for item in ddg_results:
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "url": item.get("href", ""),
                }
            )
        logger.info(f"Búsqueda DuckDuckGo exitosa para: '{query}' ({len(results)} resultados)")
    except Exception as e:
        logger.error(f"Error en búsqueda DuckDuckGo ({e}) para query: '{query}'")

    return results


async def async_search_musicology(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Ejecuta la búsqueda de forma no bloqueante en el threadpool de asyncio."""
    return await asyncio.to_thread(search_musicology, query, max_results)


def format_search_evidence(search_results: List[Dict[str, str]]) -> Tuple[str, List[str]]:
    """
    Formatea los resultados de búsqueda para inyectarlos en el prompt del LLM y
    devuelve también los fragmentos crudos estructurados para la trazabilidad en audit_track.
    """
    raw_snippets = []
    formatted_parts = []
    for idx, r in enumerate(search_results, start=1):
        snippet_clean = re.sub(r"\s+", " ", r.get("snippet", "")).strip()
        title_clean = r.get("title", "").strip()
        url = r.get("url", "").strip()
        entry = f"[{idx}] {title_clean} - {snippet_clean} (Fuente: {url})"
        raw_snippets.append(entry)
        formatted_parts.append(f"Fuente [{idx}] {title_clean}:\n{snippet_clean}\nURL: {url}")

    prompt_text = "\n\n".join(formatted_parts) if formatted_parts else "No se encontraron fuentes externas."
    return prompt_text, raw_snippets


def validate_track_guide_deterministically(
    guide_data: Any,
    duration_seconds: int,
) -> Tuple[bool, Optional[str], bool, bool]:
    """
    Valida determinísticamente las restricciones de tiempo y solapamiento.
    Devuelve: (is_valid, error_message, overlap_detected, duration_exceeded)
    """
    milestones = getattr(guide_data, "milestones", [])
    if not milestones:
        return False, "La guía no contiene ningún hito musical (milestones).", False, False

    # 1. Verificar límites de duración
    for m in milestones:
        if m.end_second > duration_seconds:
            err = (
                f"El hito '{m.section_name}' finaliza en el segundo {m.end_second}, "
                f"lo cual excede la duración total de la pista ({duration_seconds} segundos)."
            )
            return False, err, False, True

        if m.start_second >= m.end_second:
            err = (
                f"El hito '{m.section_name}' tiene start_second ({m.start_second}) >= "
                f"end_second ({m.end_second})."
            )
            return False, err, False, False

    # 2. Verificar solapamiento estricto (ordenando por start_second)
    sorted_milestones = sorted(milestones, key=lambda x: x.start_second)
    for i in range(len(sorted_milestones) - 1):
        current_m = sorted_milestones[i]
        next_m = sorted_milestones[i + 1]
        # Si el fin del actual es estrictamente mayor que el inicio del siguiente
        if current_m.end_second > next_m.start_second:
            err = (
                f"Solapamiento detectado: El hito '{current_m.section_name}' ({current_m.start_second}-{current_m.end_second}s) "
                f"se solapa con '{next_m.section_name}' ({next_m.start_second}-{next_m.end_second}s)."
            )
            return False, err, True, False

    return True, None, False, False


# --- Wrapper con Exponential Backoff para Rate Limits (Regla 1) ---

@retry(
    retry=retry_if_exception(is_rate_limit_or_transient_error),
    wait=wait_exponential(multiplier=2, min=3, max=45),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _execute_with_retry(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Ejecuta una función asíncrona con reintentos exponenciales ante 429 / ResourceExhausted."""
    return await func(*args, **kwargs)


async def invoke_with_rate_limit_resilience(
    coroutine_func: Callable,
    *args: Any,
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs: Any,
) -> Any:
    """
    Ejecuta una llamada asíncrona al LLM protegiéndola con:
    1. asyncio.Semaphore para limitar concurrencia (ej. max 3 llamadas simultáneas).
    2. Reintentos exponenciales (tenacity) si ocurre un 429 ResourceExhausted o RateLimit.
    """
    if semaphore:
        async with semaphore:
            return await _execute_with_retry(coroutine_func, *args, **kwargs)
    return await _execute_with_retry(coroutine_func, *args, **kwargs)
