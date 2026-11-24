"""
src/graph.py - Orquestador LangGraph con patrón Fan-Out / Quality Gate / Fan-In.
Implementa:
1. Concurrencia controlada con asyncio.Semaphore(3) y reintentos ante Rate Limits.
2. Nodo research_track: busca evidencia musicológica real, guarda raw_search_results y genera TrackGuide accesible para piano.
3. Nodo audit_track: Quality Gate determinista + LLM Auditor independiente (temp=0) contrastando contra raw_search_results.
4. Degradación elegante tras 3 intentos sin detener el pipeline.
5. Nodo fan_in_compile: ordena las pistas y compila el HTML final autocontenido.
"""

import asyncio
import logging
import operator
import os
from typing import Any, Callable, Dict, List, Optional, TypedDict, Annotated

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.extractor import AppleMusicExtractor
from src.schemas import (
    AlbumMetadata,
    AuditResult,
    MusicalMilestone,
    TrackGuide,
    TrackInfo,
)
from src.tools import (
    async_search_musicology,
    format_search_evidence,
    invoke_with_rate_limit_resilience,
    validate_track_guide_deterministically,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Semáforo global por defecto (1 llamada concurrente a Gemini para máxima estabilidad ante picos de demanda gratuita)
DEFAULT_SEMAPHORE = asyncio.Semaphore(1)


# --- Estados del Grafo (TypedDict para compatibilidad nativa con LangGraph) ---

class TrackWorkflowState(TypedDict):
    """Estado individual para el procesamiento y auditoría de una pista."""
    album_metadata: AlbumMetadata
    track: TrackInfo
    attempt: int
    previous_feedback: Optional[str]
    raw_search_results: List[str]
    guide: Optional[TrackGuide]
    audit: Optional[AuditResult]
    approved: bool
    finished: bool


class OverallGraphState(TypedDict):
    """Estado global del pipeline completo."""
    album_url: str
    metadata: Optional[AlbumMetadata]
    tracks: List[TrackInfo]
    # Reductor operator.add para agregar las guías generadas en paralelo por Send()
    guides: Annotated[List[TrackGuide], operator.add]
    final_html: Optional[str]
    errors: Annotated[List[str], operator.add]


# --- Factoría de Modelos y Nodos ---

def get_default_llm(temperature: float = 0.0, model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Instancia el modelo de Gemini configurado (por defecto gemini-3.5-flash) con API Key y timeout."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    selected_model = model_name or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
    return ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=temperature,
        api_key=api_key,
        timeout=90,
    )


def create_track_subgraph(
    researcher_llm: Optional[Any] = None,
    auditor_llm: Optional[Any] = None,
    search_func: Optional[Callable] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    model_name: Optional[str] = None,
):
    """
    Construye el subgrafo de procesamiento por pista:
    research_track -> audit_track -> [Aprobado / Reintento / Degradación elegante tras max_attempts]
    """
    sem = semaphore or DEFAULT_SEMAPHORE
    search_callable = search_func or async_search_musicology

    # Si no se pasan LLMs, instanciamos el modelo especificado
    res_llm = researcher_llm or get_default_llm(temperature=0.2, model_name=model_name)
    aud_llm = auditor_llm or get_default_llm(temperature=0.0, model_name=model_name)

    # Parser estructurado para Pydantic
    structured_researcher = res_llm.with_structured_output(TrackGuide)
    structured_auditor = aud_llm.with_structured_output(AuditResult)

    async def research_track_node(state: TrackWorkflowState) -> Dict[str, Any]:
        track = state["track"]
        meta = state["album_metadata"]
        attempt = state["attempt"]
        feedback = state["previous_feedback"]

        logger.info(f"Iniciando investigación para pista {track.track_number}: '{track.title}' (Intento {attempt}/{max_attempts})")

        # 1. Búsqueda musicológica real
        query = f"{meta.composer} {track.title} {track.movement or ''} piano analysis form structure"
        search_items = await search_callable(query, max_results=5)
        formatted_evidence, raw_snippets = format_search_evidence(search_items)

        # 2. Prompt pedagógico para aficionado de piano (1-2 años)
        system_prompt = (
            "Eres un musicólogo y pedagogo de piano de renombre internacional. "
            "Tu objetivo es crear una Guía de Escucha Musical accesible y fascinante para un aficionado "
            "con 1-2 años de experiencia tocando el piano. "
            "\nREGLAS FUNDAMENTALES:\n"
            "1. Tono pedagógico claro, musical y visual. Evita jerga académica excesiva o abstracta. Si usas un término técnico, explícalo.\n"
            "2. En cada hito (milestone), explica claramente qué escuchar, qué hace la mano derecha (melodía, fraseo, articulación) "
            "y qué hace la mano izquierda (acompañamiento, arpegios, bajos de Alberti, soporte armónico).\n"
            "3. LÍMITES TEMPORALES ESTRICTOS: La duración total de la pista es de EXACTAMENTE {duration} segundos. "
            "Ningún hito puede terminar después del segundo {duration}. "
            "Los hitos deben ser cronológicos y NO se pueden solapar en tiempo (end_second de un hito <= start_second del siguiente).\n"
            "4. Basa tus análisis y marcas de tiempo en la evidencia recuperada y hechos reales de la obra. Cita en source_evidence las fuentes.\n"
            "5. Responde estrictamente con el esquema JSON requerido."
        ).format(duration=track.duration_seconds)

        human_prompt_parts = [
            f"OBRA: {track.title}",
            f"MOVIMIENTO: {track.movement or 'Único'}",
            f"COMPOSITOR: {meta.composer}",
            f"INTÉRPRETE: {meta.performer}",
            f"DURACIÓN TOTAL EXACTA: {track.duration_seconds} segundos.",
            f"\n--- EVIDENCIA Y FUENTES RECUPERADAS ---\n{formatted_evidence}",
        ]

        if feedback:
            human_prompt_parts.append(
                f"\n⚠️ ATENCIÓN: El auditor rechazó el intento anterior con las siguientes observaciones que DEBES corregir obligatoriamente:\n{feedback}"
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(human_prompt_parts)),
        ]

        # Invocación con semáforo y resiliencia ante Rate Limits (Regla 1)
        guide_result: TrackGuide = await invoke_with_rate_limit_resilience(
            structured_researcher.ainvoke,
            messages,
            semaphore=sem,
        )

        # Asegurar metadatos base consistentes
        guide_result.track_number = track.track_number
        guide_result.title = track.title
        guide_result.duration_seconds = track.duration_seconds

        return {
            "guide": guide_result,
            "raw_search_results": raw_snippets,  # Guardar para trazabilidad en audit_track (Regla 2)
        }

    async def audit_track_node(state: TrackWorkflowState) -> Dict[str, Any]:
        track = state["track"]
        meta = state["album_metadata"]
        guide = state["guide"]
        raw_evidence = state.get("raw_search_results", [])
        attempt = state["attempt"]

        logger.info(f"Auditor Quality Gate para pista {track.track_number} (Intento {attempt}/{max_attempts})")

        # Paso 1: Validación determinista matemática (cero gasto de tokens)
        is_valid, error_msg, overlap, exceeded = validate_track_guide_deterministically(
            guide,
            duration_seconds=track.duration_seconds,
        )

        if not is_valid:
            audit_res = AuditResult(
                approved=False,
                feedback=f"Rechazo Determinista: {error_msg}",
                overlap_detected=overlap,
                duration_exceeded=exceeded,
                attempt_number=attempt,
            )
            return {"audit": audit_res, "approved": False}

        # Paso 2: Auditoría musicológica con LLM temp=0 contrastando contra fragmentos crudos (Regla 2)
        audit_system_prompt = (
            "Eres un Auditor de Control de Calidad Musicológico (Quality Gate) implacable y con temperature=0. "
            "Tu misión es verificar la veracidad y coherencia de la Guía de Escucha generada contra los fragmentos reales recuperados. "
            "\nCRITERIOS DE AUDITORÍA:\n"
            "1. Contrasta las secciones y afirmaciones contra las fuentes originales provistas. "
            "Está ESTRICTAMENTE PROHIBIDO inventar movimientos, modulaciones o hitos que contradigan la realidad de la obra.\n"
            "2. Verifica que las descripciones de mano derecha y mano izquierda correspondan a la escritura pianística real.\n"
            "3. Si todo es sólido, aprueba con approved=True. Si hay alucinaciones o datos no respaldados, rechaza con approved=False "
            "y da retroalimentación concisa y accionable para corregir los hitos."
        )

        milestones_summary = "\n".join(
            [
                f"- [{m.timestamp_label}] {m.section_name} | MD: {m.right_hand} | MI: {m.left_hand}"
                for m in guide.milestones
            ]
        )

        raw_evidence_text = "\n".join(raw_evidence) if raw_evidence else "Sin fragmentos externos."

        audit_human_prompt = (
            f"OBRA: {track.title} ({meta.composer})\n"
            f"DURACIÓN: {track.duration_seconds}s\n\n"
            f"--- FUENTES ORIGINALES RECUPERADAS (RAW SEARCH RESULTS) ---\n{raw_evidence_text}\n\n"
            f"--- GUÍA GENERADA POR EL INVESTIGADOR ---\n"
            f"Contexto: {guide.historical_context}\n"
            f"Estructura: {guide.structure_overview}\n"
            f"Hitos:\n{milestones_summary}\n"
            f"Citas reportadas: {guide.source_evidence}"
        )

        messages = [
            SystemMessage(content=audit_system_prompt),
            HumanMessage(content=audit_human_prompt),
        ]

        llm_audit: AuditResult = await invoke_with_rate_limit_resilience(
            structured_auditor.ainvoke,
            messages,
            semaphore=sem,
        )
        llm_audit.attempt_number = attempt

        return {"audit": llm_audit, "approved": llm_audit.approved}

    def route_after_audit(state: TrackWorkflowState) -> str:
        attempt = state["attempt"]
        approved = state["approved"]

        if approved:
            logger.info(f"✅ Pista {state['track'].track_number} aprobada con éxito en el intento {attempt}.")
            return END

        if attempt >= max_attempts:
            logger.warning(
                f"⚠️ Pista {state['track'].track_number} agotó {max_attempts} intentos sin aprobar el Quality Gate. "
                "Activando nodo de degradación elegante..."
            )
            return "degrade_gracefully"

        logger.info(f"🔄 Pista {state['track'].track_number} rechazada en intento {attempt}. Pasando a reintento...")
        return "prepare_retry"

    def prepare_retry_node(state: TrackWorkflowState) -> Dict[str, Any]:
        """Nodo que incrementa el contador de intentos y propaga el feedback del auditor."""
        current_attempt = state["attempt"]
        audit = state.get("audit")
        feedback = audit.feedback if audit else "Rechazado en auditoría."
        return {
            "attempt": current_attempt + 1,
            "previous_feedback": feedback,
        }

    def degrade_gracefully_node(state: TrackWorkflowState) -> Dict[str, Any]:
        """Nodo de degradación elegante: marca warning y observaciones sin detener el álbum."""
        guide = state.get("guide")
        audit = state.get("audit")
        feedback = audit.feedback if audit else f"No aprobó auditoría tras {max_attempts} intentos."
        if guide:
            guide.audit_warning = True
            guide.audit_feedback = feedback
        return {
            "guide": guide,
            "finished": True,
        }

    subgraph_builder = StateGraph(TrackWorkflowState)
    subgraph_builder.add_node("research_track", research_track_node)
    subgraph_builder.add_node("audit_track", audit_track_node)
    subgraph_builder.add_node("prepare_retry", prepare_retry_node)
    subgraph_builder.add_node("degrade_gracefully", degrade_gracefully_node)

    subgraph_builder.add_edge(START, "research_track")
    subgraph_builder.add_edge("research_track", "audit_track")
    subgraph_builder.add_conditional_edges(
        "audit_track",
        route_after_audit,
        {
            "prepare_retry": "prepare_retry",
            "degrade_gracefully": "degrade_gracefully",
            END: END,
        },
    )
    subgraph_builder.add_edge("prepare_retry", "research_track")
    subgraph_builder.add_edge("degrade_gracefully", END)

    return subgraph_builder.compile()


# --- Grafo Global con Fan-Out / Fan-In ---

def create_listening_guide_graph(
    researcher_llm: Optional[Any] = None,
    auditor_llm: Optional[Any] = None,
    search_func: Optional[Callable] = None,
    extractor: Optional[AppleMusicExtractor] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    model_name: Optional[str] = None,
):
    """
    Construye el grafo general con patrón:
    1. extract_metadata (Descarga metadatos y pistas con AppleMusicExtractor).
    2. fan_out_tracks (Send() en paralelo para cada pista).
    3. process_single_track (Ejecuta el subgrafo de investigación + Quality Gate con reintentos).
    4. fan_in_compile (Junta todas las pistas aprobadas/degradadas y renderiza HTML).
    """
    sem = semaphore or DEFAULT_SEMAPHORE
    music_extractor = extractor or AppleMusicExtractor()

    # Compilamos el subgrafo por pista
    track_subgraph = create_track_subgraph(
        researcher_llm=researcher_llm,
        auditor_llm=auditor_llm,
        search_func=search_func,
        semaphore=sem,
        max_attempts=max_attempts,
        model_name=model_name,
    )

    async def extract_metadata_node(state: OverallGraphState) -> Dict[str, Any]:
        if state.get("metadata") and state.get("tracks"):
            logger.info("Metadatos y pistas preseleccionadas detectadas en el estado inicial.")
            return {
                "metadata": state["metadata"],
                "tracks": state["tracks"],
            }
        url = state["album_url"]
        logger.info(f"Extrayendo metadatos del álbum desde: {url}")
        metadata = await asyncio.to_thread(music_extractor.extract, url)
        return {
            "metadata": metadata,
            "tracks": metadata.tracks,
        }

    def fan_out_tracks(state: OverallGraphState) -> List[Send]:
        metadata = state["metadata"]
        tracks = state.get("tracks") or (metadata.tracks if metadata else [])
        logger.info(f"Lanzando Fan-Out para {len(tracks)} pistas con concurrencia controlada (Semáforo)")

        return [
            Send(
                "process_single_track",
                {
                    "album_metadata": metadata,
                    "track": t,
                    "attempt": 1,
                    "previous_feedback": None,
                    "raw_search_results": [],
                    "guide": None,
                    "audit": None,
                    "approved": False,
                    "finished": False,
                },
            )
            for t in tracks
        ]

    async def process_single_track_node(state: TrackWorkflowState) -> Dict[str, Any]:
        """Procesa una pista individual invocando el subgrafo con Quality Gate."""
        track_num = state["track"].track_number
        try:
            result = await track_subgraph.ainvoke(state)
            guide = result.get("guide")
            if guide:
                return {"guides": [guide]}
            return {"errors": [f"Pista {track_num} no produjo una guía válida."]}
        except Exception as e:
            logger.error(f"Error procesando pista {track_num}: {e}")
            return {"errors": [f"Error inesperado en pista {track_num}: {str(e)}"]}

    def fan_in_compile_node(state: OverallGraphState) -> Dict[str, Any]:
        """Ordena las pistas completadas y compila el archivo HTML final."""
        meta = state.get("metadata")
        guides: List[TrackGuide] = state.get("guides", [])
        tracks = state.get("tracks") or []

        # Ordenar cronológicamente por número de pista
        sorted_guides = sorted(guides, key=lambda g: g.track_number)
        logger.info(f"Fan-In Compilación: Compilando {len(sorted_guides)} pistas ordenadas.")

        # Deducir el nombre específico y formal de la obra analizada (ej. compositor + obra + tonalidad)
        work_names = list(dict.fromkeys([t.work_name for t in tracks if getattr(t, "work_name", None)]))
        if len(work_names) == 1 and work_names[0]:
            work_title = f"{meta.composer}: {work_names[0]}" if meta else work_names[0]
        else:
            work_title = meta.title if meta else "Guía de Escucha Clásica"

        # Intentar cargar plantilla Jinja2 si existe, de lo contrario generar HTML estructurado autocontenido
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
        template_file = os.path.join(template_dir, "guide_template.html")

        if os.path.exists(template_file):
            env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(["html", "xml"]),
            )
            template = env.get_template("guide_template.html")
            html_content = template.render(
                album=meta,
                work_title=work_title,
                guides=sorted_guides,
                total_tracks=len(sorted_guides),
            )
        else:
            html_content = render_fallback_html(meta, sorted_guides)

        return {"final_html": html_content}

    builder = StateGraph(OverallGraphState)
    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("process_single_track", process_single_track_node)
    builder.add_node("fan_in_compile", fan_in_compile_node)

    builder.add_edge(START, "extract_metadata")
    builder.add_conditional_edges("extract_metadata", fan_out_tracks, ["process_single_track"])
    builder.add_edge("process_single_track", "fan_in_compile")
    builder.add_edge("fan_in_compile", END)

    return builder.compile()


def render_fallback_html(meta: Optional[AlbumMetadata], guides: List[TrackGuide]) -> str:
    """Generador HTML de contingencia por si la plantilla aún no está creada."""
    title = meta.title if meta else "Guía de Escucha Clásica"
    composer = meta.composer if meta else "Compositor"
    performer = meta.performer if meta else "Intérprete"

    sections = []
    for g in guides:
        milestone_items = "".join(
            f"<li><strong>[{m.timestamp_label}] {m.section_name}</strong>: {m.what_to_listen} "
            f"<br><em>MD:</em> {m.right_hand} | <em>MI:</em> {m.left_hand}</li>"
            for m in g.milestones
        )
        warning_tag = f"<p style='color:orange;'>⚠️ Advertencia: {g.audit_feedback}</p>" if g.audit_warning else ""
        sections.append(
            f"<div class='track-guide'><h3>Pista {g.track_number}: {g.title}</h3>"
            f"{warning_tag}<p>{g.historical_context}</p><p><strong>Estructura:</strong> {g.structure_overview}</p>"
            f"<ul>{milestone_items}</ul></div>"
        )

    return f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8"><title>{title}</title></head>
<body>
<h1>{title}</h1>
<h2>{composer} - {performer}</h2>
{''.join(sections)}
</body>
</html>"""
