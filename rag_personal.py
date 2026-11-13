import asyncio
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # carga ANTHROPIC_API_KEY desde .env

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

# 1) Modelos: Claude para razonar, embeddings locales (multilingües) para buscar
Settings.llm = Anthropic(model="claude-sonnet-5-5", max_tokens=2048)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
MANIFEST = STORAGE_DIR / "manifest.json"

# Tipos de archivo que el RAG sabe leer
FORMATOS = {
    ".pdf", ".docx",                 # documentos
    ".pptx", ".pptm",                # PowerPoint (el .ppt antiguo no se puede leer)
    ".csv", ".xlsx", ".xls",         # tablas
    ".epub",                         # libros
    ".ipynb",                        # notebooks de Jupyter
    ".txt", ".md", ".json", ".html", # texto plano
}


class LectorCSV(BaseReader):
    """Cada fila como 'columna: valor, ...' para que Claude sepa qué es cada dato
    (el lector por defecto descarta los encabezados)."""

    def load_data(self, file, extra_info=None, fs=None):
        import pandas as pd

        df = pd.read_csv(file, sep=None, engine="python", dtype=str).fillna("")
        filas = [
            ", ".join(f"{col}: {val}" for col, val in fila.items())
            for _, fila in df.iterrows()
        ]
        return [Document(text="\n".join(filas), metadata=extra_info or {})]


class LectorNotebook(BaseReader):
    """Celdas de texto, código y salidas de un notebook de Jupyter
    (el lector por defecto ignora los notebooks que no se han ejecutado)."""

    def load_data(self, file, extra_info=None, fs=None):
        nb = json.loads(Path(file).read_text("utf-8"))
        partes = []
        for celda in nb.get("cells", []):
            texto = "".join(celda.get("source", ""))
            if celda["cell_type"] == "code":
                texto = f"```python\n{texto}\n```"
                for salida in celda.get("outputs", []):
                    resultado = salida.get("text") or salida.get("data", {}).get("text/plain")
                    if resultado:
                        texto += "\nSalida:\n" + "".join(resultado)
            if texto.strip():
                partes.append(texto)
        return [Document(text="\n\n".join(partes), metadata=extra_info or {})]


def archivos_en_data() -> dict[str, list[int]]:
    """Huella de data/: ruta -> [tamaño, fecha de modificación].
    Si cambia, es que se añadió, modificó o borró algún documento."""
    huella = {}
    for f in sorted(DATA_DIR.rglob("*")):
        if (
            f.is_file()
            and f.suffix.lower() in FORMATOS
            and not f.name.startswith(("~$", "."))  # temporales de Office y ocultos
        ):
            st = f.stat()
            huella[f.relative_to(DATA_DIR).as_posix()] = [st.st_size, st.st_mtime_ns]
    return huella


def describir_cambios(antes: dict, ahora: dict) -> None:
    for f in ahora.keys() - antes.keys():
        print(f"  + nuevo:      {f}")
    for f in antes.keys() - ahora.keys():
        print(f"  - eliminado:  {f}")
    for f in ahora.keys() & antes.keys():
        if ahora[f] != antes[f]:
            print(f"  * modificado: {f}")


# 2) Indexar solo si data/ cambió desde la última vez; si no, cargar desde disco
huella_actual = archivos_en_data()
huella_guardada = json.loads(MANIFEST.read_text("utf-8")) if MANIFEST.exists() else None

if huella_guardada == huella_actual:
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
else:
    if not huella_actual:
        raise SystemExit(
            f"No hay documentos compatibles en '{DATA_DIR}/'. "
            f"Formatos aceptados: {', '.join(sorted(FORMATOS))}"
        )
    print("Cambios detectados en data/, reindexando...")
    if huella_guardada is not None:
        describir_cambios(huella_guardada, huella_actual)

    documents = SimpleDirectoryReader(
        input_files=[str(DATA_DIR / f) for f in huella_actual],
        file_extractor={".csv": LectorCSV(), ".ipynb": LectorNotebook()},
    ).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    shutil.rmtree(STORAGE_DIR, ignore_errors=True)
    index.storage_context.persist(str(STORAGE_DIR))
    MANIFEST.write_text(json.dumps(huella_actual, indent=2), "utf-8")
    print(f"Indexados {len(huella_actual)} archivo(s).\n")

query_engine = index.as_query_engine(similarity_top_k=5)


# 3) Herramienta de búsqueda que usará el agente
async def buscar_en_mis_documentos(pregunta: str) -> str:
    """Busca en los documentos personales del usuario (PDFs, Word, PowerPoint,
    tablas CSV/Excel, libros EPUB, notebooks, notas) y devuelve la información
    relevante para responder la pregunta."""
    respuesta = await query_engine.aquery(pregunta)
    return str(respuesta)


# 4) Agente con Claude
agent = FunctionAgent(
    tools=[buscar_en_mis_documentos],
    llm=Settings.llm,
    system_prompt=(
        "Eres mi asistente personal. Responde en español. Para cualquier pregunta "
        "sobre mi información, usa la herramienta buscar_en_mis_documentos. "
        "Si no encuentras la respuesta en los documentos, dilo claramente."
    ),
)


# 5) Chat en terminal con memoria de la conversación
async def main():
    ctx = Context(agent)
    print("RAG personal listo. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        respuesta = await agent.run(user_msg=pregunta, ctx=ctx)
        print(f"\nClaude: {respuesta}\n")


if __name__ == "__main__":
    asyncio.run(main())
