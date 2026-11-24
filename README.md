# Classical Listening Guide Generator 🎼

Pipeline multi-agente automatizado en Python que procesa álbumes de música clásica de **Apple Music** y genera una **Guía de Escucha Interactiva en un único archivo HTML 100 % autocontenido y offline** (sin CDNs, sin Google Fonts, sin dependencias externas).

---

## 🏛️ Arquitectura del Sistema

El pipeline utiliza **LangGraph** implementando un patrón distribuido **Fan-Out / Quality Gate / Fan-In**:

```
[Apple Music URL]
       │
       ▼
[src/extractor.py] ──> Extracción determinista (JSON-LD Schema.org, sin LLM)
       │
       ▼ (LangGraph Fan-Out con Send())
┌────────────────────────────────────────────────────────────────────────┐
│  POR CADA PISTA (en paralelo, con asyncio.Semaphore(3))                │
│                                                                        │
│  [Nodo research_track] (Gemini 3.5 Flash)                              │
│   ├── Búsqueda musicológica real (Tavily con fallback a DuckDuckGo)    │
│   ├── Genera TrackGuide pedagógico (Mano Derecha vs Mano Izquierda)    │
│   └── Guarda raw_search_results para auditoría independiente           │
│         │                                                              │
│         ▼                                                              │
│  [Nodo audit_track (Quality Gate)]                                     │
│   ├── Validación determinista (tiempos <= duración, sin solapamientos) │
│   └── Auditor LLM (temp=0) contrastando contra raw_search_results      │
│         │                                                              │
│         ├──> APROBADO ──> Pasa al fan-in                               │
│         ├──> RECHAZADO (< 3 intentos) ──> Reencola con feedback        │
│         └──> RECHAZADO (>= 3 intentos) ──> Degradación elegante        │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼ (LangGraph Fan-In)
[Nodo fan_in_compile]
       │
       ▼
[templates/guide_template.html] ──> Renderizado Jinja2 offline (Cero CDNs)
       │
       ▼
[guia_escucha.html]
```

El diagrama Mermaid completo y estilizado se encuentra en [docs/graph.mmd](docs/graph.mmd).

---

## 🚀 Requisitos e Instalación

* **Python 3.12+**
* Entorno virtual activo con las dependencias:

```bash
pip install -r requirements.txt
```

### Configuración de Variables de Entorno (`.env`)

Crea o edita tu archivo `.env` en la raíz del proyecto:

```env
GEMINI_API_KEY=tu_api_key_de_gemini
TAVILY_API_KEY=tu_api_key_de_tavily # Opcional: Si no se provee, usa DuckDuckGo automáticamente
```

---

## 💻 Uso del CLI (`main.py`)

```bash
python main.py <URL_APPLE_MUSIC> [-o OUTPUT_HTML] [-t PISTAS] [-m MODELO] [--max-attempts MAX] [-v]
```

### Ejemplo:

```bash
python main.py "https://music.apple.com/es/album/beethoven-piano-sonatas-nos-8-14-23/1452538123" -o guia_beethoven.html
```

### Argumentos:
* `url`: URL completa del álbum en Apple Music.
* `-o, --output`: Ruta del archivo HTML de salida (por defecto: `guia_escucha.html`).
* `-t, --tracks`: Pistas a procesar, separadas por comas o rangos (ej. `1,2,3` o `1-3`). Si se omite, procesa todo el álbum.
* `-m, --model`: Modelo de Gemini a utilizar (por defecto: `gemini-3.5-flash`).
* `--max-attempts`: Número máximo de intentos ante rechazos del Quality Gate antes de degradar con advertencia (por defecto: `3`).
* `-v, --verbose`: Muestra logs detallados del pipeline.

---

## 🎧 Características de la Guía HTML Generada

1. **100 % Offline y Autocontenida**: Cero llamadas de red externas, fuentes seguras del sistema y CSS/JS inline empaquetado.
2. **Cronómetro Sincronizador en Tiempo Real**:
   * Controles: `Play`, `Pausa`, `Reset`, `-5s` y `+5s`.
   * Permite sincronizar manualmente la lectura de la guía mientras reproduces el álbum en Apple Music.
3. **Resaltado y Auto-Scroll de Hitos**:
   * A medida que el cronómetro avanza, resalta automáticamente el hito activo y hace scroll suave al compás correspondiente.
4. **Enfoque Pedagógico para Piano**:
   * Desglose claro por mano:
     * **Mano Derecha (Melodía / Textura)**: Fraseo, articulación, dinámica.
     * **Mano Izquierda (Acompañamiento / Armonía)**: Bajos de Alberti, acordes, soporte armónico.
     * **Consejo práctico al piano**: Recomendaciones de escucha y pedal.
5. **Selector de Pistas**: Pestañas de navegación instantánea entre todos los movimientos y obras del álbum.

---

## 🧪 Ejecución de Pruebas Unitarias

```bash
python -m pytest tests/
```
