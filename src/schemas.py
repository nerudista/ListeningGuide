"""
src/schemas.py - Modelos de datos Pydantic v2 para el pipeline Classical Listening Guide.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class TrackInfo(BaseModel):
    """Información básica de una pista extraída del álbum."""
    track_number: int = Field(..., description="Número de pista dentro del álbum (1-indexed)")
    title: str = Field(..., description="Título completo de la pista tal como aparece en Apple Music")
    duration_seconds: int = Field(..., ge=1, description="Duración exacta de la pista en segundos")
    work_name: Optional[str] = Field(None, description="Nombre de la obra general (ej. Sonata No. 14 Op. 27 No. 2)")
    movement: Optional[str] = Field(None, description="Nombre del movimiento o tempo (ej. I. Adagio sostenuto)")
    composer: Optional[str] = Field(None, description="Compositor de la pista")
    performer: Optional[str] = Field(None, description="Intérprete o solista")
    url: Optional[str] = Field(None, description="URL directa a la pista si está disponible")


class AlbumMetadata(BaseModel):
    """Metadatos completos del álbum extraídos de Apple Music."""
    title: str = Field(..., description="Título del álbum")
    composer: str = Field(..., description="Compositor principal")
    performer: str = Field(..., description="Intérprete, orquesta o pianista principal")
    label: Optional[str] = Field(None, description="Sello discográfico (ej. Deutsche Grammophon)")
    release_year: Optional[int] = Field(None, description="Año de lanzamiento o grabación")
    album_url: str = Field(..., description="URL original del álbum en Apple Music")
    tracks: List[TrackInfo] = Field(default_factory=list, description="Lista ordenada de pistas del álbum")

    @property
    def total_duration_seconds(self) -> int:
        return sum(t.duration_seconds for t in self.tracks)


class MusicalMilestone(BaseModel):
    """Hito musical accesible con marcas de tiempo e indicaciones prácticas."""
    start_second: int = Field(..., ge=0, description="Segundo de inicio del hito dentro de la pista")
    end_second: int = Field(..., ge=1, description="Segundo final del hito dentro de la pista")
    timestamp_label: str = Field(
        ...,
        description="Etiqueta legible del tramo temporal (ej. '00:00 - 00:45' o '01:20 - 02:15')"
    )
    section_name: str = Field(
        ...,
        description="Nombre formal o pedagógico de la sección (ej. 'Tema Principal / Exposición', 'Desarrollo agitado')"
    )
    what_to_listen: str = Field(
        ...,
        description=(
            "Guía pedagógica clara para un estudiante aficionado con 1-2 años de piano. "
            "Explica el carácter emocional, cambios dinámicos y eventos sonoros clave sin jerga académica excesiva."
        )
    )
    right_hand: str = Field(
        ...,
        description=(
            "Qué hace la mano derecha / melodía en este tramo: fraseo, saltos, textura melódica, "
            "articulación (legato/staccato) y registro."
        )
    )
    left_hand: str = Field(
        ...,
        description=(
            "Qué hace la mano izquierda / acompañamiento: tipo de acompañamiento (bajos de Alberti, arpegios, "
            "acordes sostenidos, octavas), pulso y soporte armónico fundamental."
        )
    )
    pedagogical_tip: Optional[str] = Field(
        None,
        description="Consejo práctico de escucha o visualización al piano (ej. 'Presta atención al pedal', 'Canta la línea superior')"
    )

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "MusicalMilestone":
        if self.start_second >= self.end_second:
            raise ValueError(
                f"start_second ({self.start_second}) debe ser menor que end_second ({self.end_second})"
            )
        return self


class TrackGuide(BaseModel):
    """Guía de escucha completa para una pista individual."""
    track_number: int = Field(..., description="Número de pista")
    title: str = Field(..., description="Título de la pista")
    duration_seconds: int = Field(..., ge=1, description="Duración total en segundos verificada con Apple Music")
    historical_context: str = Field(
        ...,
        description="Breve contexto histórico, circunstancia de composición o anécdota pedagógica relevante"
    )
    structure_overview: str = Field(
        ...,
        description="Resumen de la arquitectura o forma musical (ej. Forma Sonata A-B-A', Tema con variaciones)"
    )
    milestones: List[MusicalMilestone] = Field(
        default_factory=list,
        description="Secuencia cronológica de hitos que cubren la obra sin solapamientos ilegales"
    )
    source_evidence: List[str] = Field(
        default_factory=list,
        description="Citas directas o referencias concretas extraídas de las búsquedas que respaldan los tiempos y análisis"
    )
    audit_warning: bool = Field(
        False,
        description="True si la pista no superó el Quality Gate tras agotar los intentos y fue compilada con advertencia"
    )
    audit_feedback: Optional[str] = Field(
        None,
        description="Observaciones o advertencias de auditoría pendientes si audit_warning es True"
    )


class AuditResult(BaseModel):
    """Resultado de la auditoría y control de calidad (Quality Gate) sobre un TrackGuide."""
    approved: bool = Field(..., description="True si la pista pasa todos los controles deterministas y musicológicos")
    feedback: str = Field(..., description="Explicación detallada de errores encontrados o confirmación de aprobación")
    overlap_detected: bool = Field(False, description="True si dos hitos se solapan en tiempo")
    duration_exceeded: bool = Field(False, description="True si algún hito supera la duración de la pista")
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Afirmaciones o timestamps no respaldados por la evidencia bibliográfica"
    )
    attempt_number: int = Field(1, description="Número de intento de auditoría para esta pista (máx. 3)")


# --- Modelos de Estado para LangGraph ---

class TrackWorkflowInput(BaseModel):
    """Input empaquetado para el patrón Send() por pista individual."""
    album_metadata: AlbumMetadata
    track: TrackInfo
    attempt: int = 1
    previous_feedback: Optional[str] = None
    accumulated_evidence: List[str] = Field(default_factory=list)
    raw_search_results: List[str] = Field(default_factory=list)
    guide: Optional[TrackGuide] = None
    audit: Optional[AuditResult] = None


class TrackWorkflowOutput(BaseModel):
    """Resultado del subgrafo/proceso por pista."""
    track_number: int
    guide: Optional[TrackGuide] = None
    audit: AuditResult
    success: bool = False


class OverallState(BaseModel):
    """Estado global del grafo LangGraph (Fan-Out / Fan-In)."""
    album_url: str
    metadata: Optional[AlbumMetadata] = None
    guides: List[TrackGuide] = Field(default_factory=list)
    final_html: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
