"""
tests/test_graph_mocked.py - Pruebas unitarias completas simulando el LLM para verificar:
1. Ciclo exitoso en intento 1.
2. Calidad determinista (rechazo por solapamiento o exceso de tiempo sin llamar al LLM).
3. Ciclo de reintento: Intento 1 rechazado -> Reencolado -> Intento 2 aprobado.
4. Degradación elegante tras 3 intentos (Regla 3).
5. Fan-Out -> Quality Gate -> Fan-In completo.
"""

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.graph import create_track_subgraph, create_listening_guide_graph
from src.schemas import (
    AlbumMetadata,
    AuditResult,
    MusicalMilestone,
    TrackGuide,
    TrackInfo,
)


@pytest.fixture
def sample_album():
    return AlbumMetadata(
        title="Beethoven: Moonlight Sonata",
        composer="Ludwig van Beethoven",
        performer="Daniel Barenboim",
        label="DG",
        release_year=1984,
        album_url="https://music.apple.com/album/sample",
        tracks=[
            TrackInfo(
                track_number=1,
                title="Sonata No. 14: I. Adagio sostenuto",
                duration_seconds=120,
                movement="I. Adagio sostenuto",
            ),
            TrackInfo(
                track_number=2,
                title="Sonata No. 14: II. Allegretto",
                duration_seconds=90,
                movement="II. Allegretto",
            ),
        ],
    )


@pytest.fixture
def valid_guide():
    return TrackGuide(
        track_number=1,
        title="Sonata No. 14: I. Adagio sostenuto",
        duration_seconds=120,
        historical_context="Obra cumbre del periodo medio temprano.",
        structure_overview="Forma lied desarrollada con pulso constante de tresillos.",
        milestones=[
            MusicalMilestone(
                start_second=0,
                end_second=60,
                timestamp_label="00:00 - 01:00",
                section_name="Introducción y Tema Principal",
                what_to_listen="El murmullo constante de los tresillos y la entrada solemne de la melodía.",
                right_hand="Tresillos en pianissimo en registro central; melodía pausada.",
                left_hand="Bajos en octavas sostenidas actuando como pedal armónico.",
            ),
            MusicalMilestone(
                start_second=60,
                end_second=120,
                timestamp_label="01:00 - 02:00",
                section_name="Desarrollo y Cadencia",
                what_to_listen="Tensión armónica creciente con modulaciones expresivas.",
                right_hand="Fraseo más amplio con acentos delicados.",
                left_hand="Acordes arpegiados que profundizan el soporte armónico.",
            ),
        ],
        source_evidence=["Tovey Beethoven Analysis, p. 112."],
    )


def test_subgraph_successful_first_attempt(sample_album, valid_guide):
    """Test 1: Pista aprobada a la primera por el auditor."""
    async def _run():
        mock_researcher = MagicMock()
        mock_researcher_structured = MagicMock()
        mock_researcher_structured.ainvoke = AsyncMock(return_value=valid_guide)
        mock_researcher.with_structured_output.return_value = mock_researcher_structured

        mock_auditor = MagicMock()
        mock_auditor_structured = MagicMock()
        mock_auditor_structured.ainvoke = AsyncMock(
            return_value=AuditResult(approved=True, feedback="Estructura e hitos verificados correctamente.")
        )
        mock_auditor.with_structured_output.return_value = mock_auditor_structured

        mock_search = AsyncMock(return_value=[{"title": "Analysis", "snippet": "Text", "url": "http://src"}])

        subgraph = create_track_subgraph(
            researcher_llm=mock_researcher,
            auditor_llm=mock_auditor,
            search_func=mock_search,
            max_attempts=3,
        )

        initial_state = {
            "album_metadata": sample_album,
            "track": sample_album.tracks[0],
            "attempt": 1,
            "previous_feedback": None,
            "raw_search_results": [],
            "guide": None,
            "audit": None,
            "approved": False,
            "finished": False,
        }

        result = await subgraph.ainvoke(initial_state)

        assert result["approved"] is True
        assert result["attempt"] == 1
        assert result["guide"].track_number == 1
        assert result["guide"].audit_warning is False
        assert len(result["raw_search_results"]) == 1

    asyncio.run(_run())


def test_subgraph_deterministic_rejection_without_llm(sample_album):
    """Test 2: Rechazo determinista inmediato si un hito excede la duración, sin gastar llamada al auditor LLM."""
    async def _run():
        invalid_guide = TrackGuide(
            track_number=1,
            title="Sonata No. 14: I. Adagio sostenuto",
            duration_seconds=120,
            historical_context="Contexto",
            structure_overview="Estructura",
            milestones=[
                MusicalMilestone(
                    start_second=0,
                    end_second=180,  # Excede los 120s
                    timestamp_label="00:00 - 03:00",
                    section_name="Sección imposible",
                    what_to_listen="...",
                    right_hand="...",
                    left_hand="...",
                )
            ],
        )

        mock_researcher = MagicMock()
        mock_researcher_structured = MagicMock()
        mock_researcher_structured.ainvoke = AsyncMock(return_value=invalid_guide)
        mock_researcher.with_structured_output.return_value = mock_researcher_structured

        mock_auditor = MagicMock()
        mock_auditor_structured = MagicMock()
        mock_auditor_structured.ainvoke = AsyncMock()  # No debería ser llamado
        mock_auditor.with_structured_output.return_value = mock_auditor_structured

        mock_search = AsyncMock(return_value=[])

        subgraph = create_track_subgraph(
            researcher_llm=mock_researcher,
            auditor_llm=mock_auditor,
            search_func=mock_search,
            max_attempts=1,  # Termina de inmediato
        )

        initial_state = {
            "album_metadata": sample_album,
            "track": sample_album.tracks[0],
            "attempt": 1,
            "previous_feedback": None,
            "raw_search_results": [],
            "guide": None,
            "audit": None,
            "approved": False,
            "finished": False,
        }

        result = await subgraph.ainvoke(initial_state)

        assert result["approved"] is False
        assert result["audit"].duration_exceeded is True
        mock_auditor_structured.ainvoke.assert_not_called()

    asyncio.run(_run())


def test_subgraph_retry_cycle(sample_album, valid_guide):
    """Test 3: Rechazo en intento 1 -> Reintento con feedback -> Aprobación en intento 2."""
    async def _run():
        mock_researcher = MagicMock()
        mock_researcher_structured = MagicMock()
        mock_researcher_structured.ainvoke = AsyncMock(return_value=valid_guide)
        mock_researcher.with_structured_output.return_value = mock_researcher_structured

        # Auditor rechaza en intento 1 y aprueba en intento 2
        mock_auditor = MagicMock()
        mock_auditor_structured = MagicMock()
        mock_auditor_structured.ainvoke = AsyncMock(
            side_effect=[
                AuditResult(approved=False, feedback="Rechazado: Faltó detallar la mano izquierda."),
                AuditResult(approved=True, feedback="Aprobado: Mano izquierda corregida."),
            ]
        )
        mock_auditor.with_structured_output.return_value = mock_auditor_structured

        mock_search = AsyncMock(return_value=[{"title": "Ref", "snippet": "Data", "url": "http://ref"}])

        subgraph = create_track_subgraph(
            researcher_llm=mock_researcher,
            auditor_llm=mock_auditor,
            search_func=mock_search,
            max_attempts=3,
        )

        initial_state = {
            "album_metadata": sample_album,
            "track": sample_album.tracks[0],
            "attempt": 1,
            "previous_feedback": None,
            "raw_search_results": [],
            "guide": None,
            "audit": None,
            "approved": False,
            "finished": False,
        }

        result = await subgraph.ainvoke(initial_state)

        assert result["approved"] is True
        assert result["attempt"] == 2
        assert mock_researcher_structured.ainvoke.call_count == 2
        assert mock_auditor_structured.ainvoke.call_count == 2

    asyncio.run(_run())


def test_subgraph_graceful_degradation_after_max_attempts(sample_album, valid_guide):
    """Test 4: Degradación elegante tras 3 intentos (Regla 3): audit_warning=True sin romper el flujo."""
    async def _run():
        mock_researcher = MagicMock()
        mock_researcher_structured = MagicMock()
        mock_researcher_structured.ainvoke = AsyncMock(return_value=valid_guide)
        mock_researcher.with_structured_output.return_value = mock_researcher_structured

        # Auditor rechaza siempre
        mock_auditor = MagicMock()
        mock_auditor_structured = MagicMock()
        mock_auditor_structured.ainvoke = AsyncMock(
            return_value=AuditResult(approved=False, feedback="Inconsistencia persistente en compases.")
        )
        mock_auditor.with_structured_output.return_value = mock_auditor_structured

        mock_search = AsyncMock(return_value=[])

        subgraph = create_track_subgraph(
            researcher_llm=mock_researcher,
            auditor_llm=mock_auditor,
            search_func=mock_search,
            max_attempts=3,
        )

        initial_state = {
            "album_metadata": sample_album,
            "track": sample_album.tracks[0],
            "attempt": 1,
            "previous_feedback": None,
            "raw_search_results": [],
            "guide": None,
            "audit": None,
            "approved": False,
            "finished": False,
        }

        result = await subgraph.ainvoke(initial_state)

        # Debe haber finalizado ordenadamente con warning marcado
        assert result["guide"].audit_warning is True
        assert "Inconsistencia persistente" in result["guide"].audit_feedback
        assert mock_researcher_structured.ainvoke.call_count == 3

    asyncio.run(_run())


def test_full_graph_fan_out_fan_in(sample_album, valid_guide):
    """Test 5: Grafo global completo con Fan-Out de 2 pistas y Fan-In de compilación."""
    async def _run():
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = sample_album

        mock_researcher = MagicMock()
        mock_researcher_structured = MagicMock()
        mock_researcher_structured.ainvoke = AsyncMock(return_value=valid_guide)
        mock_researcher.with_structured_output.return_value = mock_researcher_structured

        mock_auditor = MagicMock()
        mock_auditor_structured = MagicMock()
        mock_auditor_structured.ainvoke = AsyncMock(
            return_value=AuditResult(approved=True, feedback="Aprobado")
        )
        mock_auditor.with_structured_output.return_value = mock_auditor_structured

        mock_search = AsyncMock(return_value=[{"title": "Note", "snippet": "Piano piece", "url": "http://src"}])

        graph = create_listening_guide_graph(
            researcher_llm=mock_researcher,
            auditor_llm=mock_auditor,
            search_func=mock_search,
            extractor=mock_extractor,
            max_attempts=2,
        )

        initial_state = {
            "album_url": "https://music.apple.com/album/sample",
            "metadata": None,
            "tracks": [],
            "guides": [],
            "final_html": None,
            "errors": [],
        }

        result = await graph.ainvoke(initial_state)

        assert result["metadata"] is not None
        assert len(result["guides"]) == 2
        assert result["final_html"] is not None
        assert "Sonata No. 14" in result["final_html"]

    asyncio.run(_run())
