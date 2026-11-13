# RAG Personal

Asistente personal en la terminal que responde preguntas sobre tus propios documentos usando **RAG** (Retrieval-Augmented Generation).

Usa **Claude** (Anthropic) para razonar y responder, **embeddings locales** multilingües (`BAAI/bge-m3`) para buscar la información relevante en tus archivos y **[graphify](https://github.com/Graphify-Labs/graphify)** para construir un grafo de conocimiento con las entidades y relaciones de tus documentos (**GraphRAG**).

## Características

- 💬 Chat en la terminal que recuerda la conversación.
- 🔍 Un agente que decide cuándo buscar en tus documentos.
- 🕸️ **GraphRAG**: además de la búsqueda vectorial, el agente consulta un grafo de conocimiento para responder preguntas sobre relaciones ("¿qué relación hay entre X e Y?").
- ⚡ Indexación incremental: solo reindexa cuando añades, modificas o eliminas archivos en `data/`.
- 🌍 Embeddings multilingües que se ejecutan en local, sin coste de API.
- 📂 Lee muchos formatos:

| Tipo | Extensiones |
|------|-------------|
| Documentos | `.pdf`, `.docx` |
| Presentaciones | `.pptx`, `.pptm` |
| Tablas | `.csv`, `.xlsx`, `.xls` |
| Libros | `.epub` |
| Notebooks | `.ipynb` |
| Texto | `.txt`, `.md`, `.json`, `.html` |

## Requisitos

- Python 3.10 o superior
- Una API key de Anthropic: [console.anthropic.com](https://console.anthropic.com/)

## Instalación

```powershell
# 1. Clonar el repositorio
git clone https://github.com/JGOM04/RAG-Personal.git
cd RAG-Personal

# 2. Crear y activar el entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate         # macOS / Linux

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` en la raíz del proyecto con tu API key:

```env
ANTHROPIC_API_KEY=tu_api_key_aqui
```

> ⚠️ El archivo `.env` está en el `.gitignore` y **nunca** debe subirse al repositorio.

## Uso

1. Crea una carpeta `data/` y copia en ella los documentos que quieras consultar (admite subcarpetas).
2. Ejecuta:

```powershell
python rag_personal.py
```

3. Haz preguntas sobre tus documentos. Escribe `salir` para terminar.

```
RAG personal listo. Escribe 'salir' para terminar.

Tú: ¿Qué temas trata el informe de marzo?

Claude: ...
```

La primera ejecución descarga el modelo de embeddings (unos 2 GB), indexa todos los documentos y construye el grafo de conocimiento, así que tarda un poco. Las siguientes cargan el índice guardado en `storage/` y el grafo de `grafo/`.

Para ver el grafo de forma interactiva, abre `grafo/graphify-out/graph.html` en el navegador.

> 💰 La construcción del grafo usa la API de Claude. Es incremental, así que solo vuelve a procesar los documentos nuevos o modificados.

## Estructura del proyecto

```
RAG-Personal/
├── rag_personal.py     # Script principal
├── requirements.txt    # Dependencias
├── .env                # API key (no se sube)
├── data/               # Tus documentos (no se sube)
├── storage/            # Índice vectorial generado (no se sube)
└── grafo/              # Grafo de conocimiento de graphify (no se sube)
```

## Cómo funciona

1. **Detección de cambios:** el script calcula una huella de `data/` (tamaño y fecha de cada archivo) y la compara con `storage/manifest.json`.
2. **Indexación:** si algo cambió, lee los documentos con LlamaIndex, genera los embeddings con `bge-m3` y guarda el índice en `storage/`.
3. **Grafo de conocimiento:** `graphify extract` usa Claude para extraer entidades (personas, proyectos, temas…) y sus relaciones, las agrupa en comunidades y guarda el grafo en `grafo/`.
4. **Agente:** un `FunctionAgent` con Claude dispone de tres herramientas:
   - `buscar_en_mis_documentos`: búsqueda vectorial que recupera los 5 fragmentos más relevantes.
   - `consultar_grafo`: recorre el grafo a partir de unas palabras clave para dar una visión de conjunto.
   - `relacion_entre`: encuentra el camino más corto entre dos conceptos del grafo.
5. **Respuesta:** Claude combina lo que encuentra, responde en español y avisa si no encuentra la información.

```
            ┌─ buscar_en_mis_documentos ─► índice vectorial (storage/)
Pregunta ─► Agente (Claude)
            └─ consultar_grafo / relacion_entre ─► grafo de conocimiento (grafo/)
```

## Tecnologías

- [LlamaIndex](https://www.llamaindex.ai/)
- [Anthropic Claude](https://www.anthropic.com/)
- [Hugging Face: BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [graphify](https://github.com/Graphify-Labs/graphify)
