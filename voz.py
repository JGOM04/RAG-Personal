"""Entrada y salida de voz del RAG personal.

- Escuchar: graba el micrófono y transcribe en local con Whisper (faster-whisper).
- Hablar: sintetiza la respuesta con voces neuronales de Microsoft (edge-tts).
"""

import io
import queue
import re

try:
    import msvcrt  # Windows: detectar Enter sin bloquear
except ImportError:
    msvcrt = None

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


def _enter_pulsado() -> bool:
    """True si se ha pulsado Enter (sin bloquear y sin dejar la tecla en el búfer)."""
    if msvcrt is None:
        return False
    pulsado = False
    while msvcrt.kbhit():
        if msvcrt.getwch() in "\r\n":
            pulsado = True
    return pulsado


def grabar(max_segundos: float = 30, silencio_final: float = 1.5,
           espera_sin_voz: float = 8) -> tuple[np.ndarray, bool]:
    """Graba del micrófono y para sola cuando dejas de hablar (o al pulsar Enter).
    Devuelve (audio a 16 kHz, si se detectó voz)."""
    frecuencia = int(sd.query_devices(kind="input")["default_samplerate"])
    bloque = int(frecuencia * 0.1)  # trozos de 100 ms
    cola: queue.Queue = queue.Queue()

    def guardar(indata, frames, tiempo, estado):
        cola.put(indata[:, 0].copy())

    _enter_pulsado()  # descarta pulsaciones previas
    print("  🎙️  Te escucho... (paro solo cuando dejes de hablar, o pulsa Enter)")
    trozos, niveles = [], []
    segundos, ultimo_con_voz, hubo_voz = 0.0, 0.0, False

    with sd.InputStream(samplerate=frecuencia, channels=1, dtype="float32",
                        blocksize=bloque, callback=guardar):
        while segundos < max_segundos:
            try:
                trozo = cola.get(timeout=2)
            except queue.Empty:
                raise RuntimeError("el micrófono no está enviando audio")
            trozos.append(trozo)
            segundos += len(trozo) / frecuencia
            nivel = float(np.sqrt(np.mean(trozo ** 2)))
            niveles.append(nivel)

            # El ruido de fondo se estima con el primer medio segundo
            ruido = float(np.median(niveles[:5]))
            if segundos > 0.5 and nivel > max(ruido * 2.5, 0.004):
                hubo_voz, ultimo_con_voz = True, segundos
            if hubo_voz and segundos - ultimo_con_voz > silencio_final:
                break
            if not hubo_voz and segundos > espera_sin_voz:
                break
            if _enter_pulsado():
                break

    audio = np.concatenate(trozos) if trozos else np.zeros(0, dtype=np.float32)
    pico = float(np.abs(audio).max()) if len(audio) else 0.0
    aviso = "" if hubo_voz else " — no detecté voz, ¿micrófono silenciado o muy bajo?"
    print(f"  (Grabados {segundos:.1f} s, volumen máximo {pico:.2f}{aviso})")
    return _a_16khz(audio, frecuencia), hubo_voz


def transcribir(audio: np.ndarray) -> str:
    if len(audio) < FRECUENCIA_WHISPER * 0.3:  # menos de 0,3 s: no hay nada que entender
        return ""
    pico = float(np.abs(audio).max())
    if pico > 0:
        audio = (audio / pico * 0.9).astype(np.float32)  # sube el volumen de micrófonos flojos
    segmentos, _ = _modelo_whisper().transcribe(
        audio, language="es", vad_filter=True, condition_on_previous_text=False
    )
    # Whisper "inventa" frases con el silencio: descarta lo que marca como no-voz
    return " ".join(s.text.strip() for s in segmentos if s.no_speech_prob < 0.6).strip()


def escuchar() -> str:
    """Graba una pregunta hablada y la devuelve como texto."""
    audio, hubo_voz = grabar()
    return transcribir(audio) if hubo_voz else ""


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
