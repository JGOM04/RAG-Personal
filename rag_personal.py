import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # carga ANTHROPIC_API_KEY desde .env
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # aviso inofensivo en Windows

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
MODELO = "claude-sonnet-5-5"
Settings.llm = Anthropic(model=MODELO, max_tokens=2048)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
MANIFEST = STORAGE_DIR / "manifest.json"
GRAFO_DIR = Path("grafo")  # salida de graphify (grafo de conocimiento)
GRAFO = GRAFO_DIR / "graphify-out" / "graph.json"

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


def graphify(*args: str, capturar: bool = False) -> subprocess.CompletedProcess:
    """Ejecuta la CLI de graphify con el Python del entorno virtual.
    Hereda ANTHROPIC_API_KEY (cargada desde .env) para usar Claude como motor."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "-m", "graphify", *args],
        env=env,
        capture_output=capturar,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def construir_grafo() -> None:
    """Extrae entidades y relaciones de data/ con graphify y guarda el grafo en grafo/.
    Es incremental: graphify solo vuelve a procesar los archivos nuevos o modificados."""
    print("Construyendo el grafo de conocimiento con graphify...")
    r = graphify("extract", str(DATA_DIR), "--backend", "claude",
                 "--model", MODELO, "--out", str(GRAFO_DIR))
    if r.returncode != 0 or not GRAFO.exists():
        print("  ! No se pudo construir el grafo; se usará solo la búsqueda vectorial.\n")
    else:
        graphify("export", "html", "--graph", str(GRAFO))
        print(f"  Grafo guardado en {GRAFO.parent} (abre graph.html en el navegador para verlo).\n")


# 2) Indexar solo si data/ cambió desde la última vez; si no, cargar desde disco
huella_actual = archivos_en_data()
huella_guardada = json.loads(MANIFEST.read_text("utf-8")) if MANIFEST.exists() else None

if huella_guardada == huella_actual:
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
    if not GRAFO.exists():
        construir_grafo()  # índice al día, pero el grafo aún no existe
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
    construir_grafo()

query_engine = index.as_query_engine(similarity_top_k=5)


# 3) Herramienta de búsqueda que usará el agente
async def buscar_en_mis_documentos(pregunta: str) -> str:
    """Busca en los documentos personales del usuario (PDFs, Word, PowerPoint,
    tablas CSV/Excel, libros EPUB, notebooks, notas) y devuelve la información
    relevante para responder la pregunta."""
    respuesta = await query_engine.aquery(pregunta)
    return str(respuesta)


# 3b) Herramientas del grafo de conocimiento (graphify)
def _salida_grafo(r: subprocess.CompletedProcess) -> str:
    salida = (r.stdout or "").strip()
    if r.returncode == 4 or not salida:
        return "El grafo no tiene nodos que coincidan con esos términos."
    return salida  # código 3 = nombre ambiguo: la salida lista los candidatos


async def consultar_grafo(terminos: str) -> str:
    """Explora el grafo de conocimiento de los documentos: entidades (personas,
    proyectos, temas, organizaciones, fechas) y cómo se relacionan entre sí.
    Úsala para preguntas sobre conexiones o visión de conjunto. Pasa palabras
    clave tal como aparecerían en los documentos, no una pregunta completa."""
    r = await asyncio.to_thread(
        graphify, "query", terminos, "--graph", str(GRAFO), "--budget", "1500", capturar=True
    )
    return _salida_grafo(r)


async def relacion_entre(concepto_a: str, concepto_b: str) -> str:
    """Encuentra el camino más corto en el grafo de conocimiento entre dos
    conceptos o entidades, para explicar cómo están relacionados."""
    r = await asyncio.to_thread(
        graphify, "path", concepto_a, concepto_b, "--graph", str(GRAFO), "--undirected",
        capturar=True,
    )
    return _salida_grafo(r)


# 4) Agente con Claude
herramientas = [buscar_en_mis_documentos]
instrucciones = (
    "Eres mi asistente personal. Responde en español. Para cualquier pregunta "
    "sobre mi información, usa la herramienta buscar_en_mis_documentos. "
)
if GRAFO.exists():
    herramientas += [consultar_grafo, relacion_entre]
    instrucciones += (
        "Para preguntas sobre relaciones entre personas, temas o proyectos, o para "
        "una visión de conjunto, usa también consultar_grafo y relacion_entre, y "
        "combina lo que encuentres con buscar_en_mis_documentos. "
    )
instrucciones += "Si no encuentras la respuesta en los documentos, dilo claramente."

agent = FunctionAgent(
    tools=herramientas,
    llm=Settings.llm,
    system_prompt=instrucciones,
)


# 5) Chat por texto o por voz, con memoria de la conversación
async def main():
    import voz

    ctx = Context(agent)
    voz_activa = True  # leer las respuestas en voz alta
    print(
        "RAG personal listo.\n"
        "  · Escribe tu pregunta, o pulsa Enter sin escribir nada para hablar.\n"
        "  · 'voz' activa o desactiva las respuestas habladas. 'salir' termina.\n"
    )
    while True:
        pregunta = input("Tú: ").strip()
        voz.callar()  # escribir o pulsar Enter interrumpe la respuesta hablada

        if not pregunta:  # Enter vacío: pregunta por voz
            try:
                pregunta = await asyncio.to_thread(voz.escuchar)
            except Exception as e:
                print(f"  ! No se pudo usar el micrófono: {e}\n")
                continue
            if not pregunta:
                print("  (No te he entendido, prueba otra vez.)\n")
                continue
            print(f"Tú (voz): {pregunta}")

        orden = pregunta.lower().strip(" .,;:!¡?¿")
        if orden in {"salir", "exit", "quit"}:
            break
        if orden == "voz":
            voz_activa = not voz_activa
            print(f"  Respuestas habladas {'activadas' if voz_activa else 'desactivadas'}.\n")
            continue

        respuesta = str(await agent.run(user_msg=pregunta, ctx=ctx))
        print(f"\nClaude: {respuesta}\n")
        if voz_activa:
            try:
                await voz.hablar(respuesta)
            except Exception as e:
                print(f"  ! No se pudo leer la respuesta en voz alta: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
