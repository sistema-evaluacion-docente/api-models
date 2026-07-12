# API Análisis de Comentarios Docentes

API de procesamiento de lenguaje natural para el análisis automatizado de comentarios de evaluación docente en español. Proyecto de tesis.

## Funcionalidades

- **Análisis de sentimiento** — clasifica comentarios como positivos, neutros o negativos ([pysentimiento](https://github.com/pysentimiento/pysentimiento)).
- **Detección de discurso de odio** — identifica contenido abusivo u ofensivo en los comentarios.
- **Clasificación de nivel de riesgo** — modelo fine-tuned (`DevOB/modelo-distilbeto-nivelDeRiesgo5`) que asigna niveles `BAJO`, `MEDIO` o `ALTO`.
- **Categorización** — clasifica comentarios en categorías predefinidas (`DevOB/modelo-distilbeto-categorias-3`).
- **Generación de resúmenes ejecutivos** — resume lotes de comentarios usando `gemma3:1b` vía Ollama, detectando menciones de acoso sexual, intimidación, violencia o abuso de poder que requieran atención institucional inmediata.
- **Rate limiting** — todos los endpoints tienen límites de frecuencia.
- **Métricas de desempeño** — cada análisis retorna tiempo de ejecución, tiempo de CPU y delta de RAM como parte de la respuesta.

## Stack

| Componente       | Tecnología                           |
| ---------------- | ------------------------------------ |
| Framework web    | FastAPI                              |
| NLP              | pysentimiento, transformers (HF)     |
| LLM              | Ollama (gemma3:1b)                   |
| Rate limiting    | slowapi                              |
| Configuración    | pydantic-settings, python-dotenv     |

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) corriendo localmente con el modelo `gemma3:1b` disponible
- Token de Hugging Face para acceder a los modelos custom (`DevOB/*`)

## Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd api.demo

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con los valores correspondientes (SUMMARIZE_PROMPT, HF_TOKEN)

# Asegurarse de que Ollama esté corriendo y tener el modelo descargado
ollama pull gemma3:1b
```

## Variables de entorno

| Variable            | Descripción                                         |
| ------------------- | --------------------------------------------------- |
| `SUMMARIZE_PROMPT`  | Prompt para la generación de resúmenes ejecutivos   |
| `HF_TOKEN`          | Token de Hugging Face para modelos privados/custom  |

## Uso

```bash
fastapi run api/main.py --port 8000
```

La API estará disponible en `http://localhost:8000`.

## Endpoints

### `GET /`

Health check.

```
curl http://localhost:8000/
```

### `POST /comments/`

Analiza un comentario individual (sentimiento, odio, riesgo, categorías).

```
curl -X POST http://localhost:8000/comments/ \
  -H "Content-Type: application/json" \
  -d '{"content": "El profesor explica muy bien los temas"}'
```

**Rate limit:** 30 peticiones/minuto

### `POST /summarize/`

Genera un resumen ejecutivo a partir de una lista de comentarios.

```
curl -X POST http://localhost:8000/summarize/ \
  -H "Content-Type: application/json" \
  -d '{"comments": ["comentario 1", "comentario 2", "comentario 3"]}'
```

**Rate limit:** 5 peticiones/minuto

## Docker

```bash
docker compose up --build
```

El contenedor expone el puerto `8000`. Límites de recursos: 2 CPUs, 6 GB RAM.

## Estructura del proyecto

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── api/
│   ├── main.py          # Aplicación FastAPI y endpoints
│   ├── nlp.py           # Inicialización de modelos NLP
│   ├── settings.py      # Configuración desde .env
│   └── models/
│       ├── comment.py   # Modelos de request (Comment, CommentList)
│       └── analysis.py  # Modelos de respuesta
```
