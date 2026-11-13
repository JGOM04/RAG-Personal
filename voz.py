"""Entrada y salida de voz del RAG personal.

- Escuchar: graba el micrófono y transcribe en local con Whisper (faster-whisper).
- Hablar: sintetiza la respuesta con voces neuronales de Microsoft (edge-tts).
"""

import io
import re

import numpy as np
import sounddevice as sd
import soundfile as sf

MODELO_WHISPER = "small"     # tiny | base | small | medium: más grande = más preciso y más lento
VOZ = "es-MX-DaliaNeural"    # otras: es-MX-JorgeNeural, es-ES-ElviraNeural, es-ES-AlvaroNeural...
FRECUENCIA_WHISPER = 16000   # Whisper trabaja con audio a 16 kHz

_whisper = None


def _modelo_whisper():
    """Carga Whisper la primera vez que se usa (la primera vez lo descarga)."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        print("  (Cargando el modelo de reconocimiento de voz...)")
        _whisper = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")
    return _whisper


def _a_16khz(audio: np.ndarray, frecuencia: int) -> np.ndarray:
    if frecuencia == FRECUENCIA_WHISPER or len(audio) == 0:
        return audio
    n = int(len(audio) * FRECUENCIA_WHISPER / frecuencia)
    x = np.linspace(0, len(audio) - 1, n)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def grabar() -> np.ndarray:
    """Graba del micrófono hasta que el usuario pulse Enter. Devuelve audio a 16 kHz."""
    frecuencia = int(sd.query_devices(kind="input")["default_samplerate"])
    bloques = []

    def guardar(indata, frames, tiempo, estado):
        bloques.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=frecuencia, channels=1, dtype="float32", callback=guardar):
        input("  🎙️  Te escucho... pulsa Enter cuando termines.")
    audio = np.concatenate(bloques) if bloques else np.zeros(0, dtype=np.float32)
    return _a_16khz(audio, frecuencia)


def transcribir(audio: np.ndarray) -> str:
    if len(audio) < FRECUENCIA_WHISPER * 0.3:  # menos de 0,3 s: no hay nada que entender
        return ""
    segmentos, _ = _modelo_whisper().transcribe(audio, language="es", vad_filter=True)
    return " ".join(s.text.strip() for s in segmentos).strip()


def escuchar() -> str:
    """Graba una pregunta hablada y la devuelve como texto."""
    return transcribir(grabar())


def _limpiar_para_voz(texto: str) -> str:
    """Quita el formato Markdown para que la voz no lea asteriscos, almohadillas, etc."""
    texto = re.sub(r"```.*?```", " (ver el código en pantalla) ", texto, flags=re.S)
    texto = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texto)   # [texto](enlace) -> texto
    texto = re.sub(r"^\s*([-*•]|\d+\.)\s+", "", texto, flags=re.M)  # viñetas y listas
    texto = re.sub(r"[*_#`>|~]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


async def sintetizar(texto: str) -> tuple[np.ndarray, int] | None:
    """Convierte texto en audio con edge-tts. Devuelve (muestras, frecuencia)."""
    import edge_tts

    texto = _limpiar_para_voz(texto)
    if not texto:
        return None
    mp3 = bytearray()
    async for trozo in edge_tts.Communicate(texto, VOZ).stream():
        if trozo["type"] == "audio":
            mp3.extend(trozo["data"])
    return sf.read(io.BytesIO(bytes(mp3)), dtype="float32")


async def hablar(texto: str) -> None:
    """Lee la respuesta en voz alta sin bloquear: se puede seguir escribiendo
    mientras habla, y callar() la interrumpe."""
    audio = await sintetizar(texto)
    if audio is not None:
        sd.play(*audio)


def callar() -> None:
    sd.stop()
