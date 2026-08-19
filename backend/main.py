import os
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import imageio_ffmpeg
from groq import Groq
from pydantic import BaseModel
import edge_tts


app = FastAPI(title="Myanmar Movie Recap AI")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# WHISPER
# =========================

MODEL_SIZE = os.getenv(
    "WHISPER_MODEL",
    "tiny"
)

whisper_model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)


# =========================
# GROQ
# =========================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)


# =========================
# REQUEST MODELS
# =========================

class RecapRequest(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "thiha"


# =========================
# HOME
# =========================

@app.get("/")
def home():

    return FileResponse(
        "index.html"
    )


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================
# TRANSCRIBE
# =========================

@app.post("/transcribe")
async def transcribe_video(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file uploaded"
        )


    input_path = None
    audio_path = None


    try:

        suffix = os.path.splitext(
            file.filename
        )[1] or ".mp4"


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(
                await file.read()
            )

            input_path = temp.name


        ffmpeg = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )


        audio_temp = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )
        )


        audio_path = audio_temp.name

        audio_temp.close()


        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                input_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                audio_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        segments, info = (
            whisper_model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True
            )
        )


        transcript = []
        timestamped_segments = []


        for segment in segments:

            text = segment.text.strip()


            if text:

                transcript.append(
                    text
                )


                timestamped_segments.append(
                    {
                        "start": round(
                            segment.start,
                            2
                        ),
                        "end": round(
                            segment.end,
                            2
                        ),
                        "text": text
                    }
                )


        return {

            "success": True,

            "filename":
                file.filename,

            "language":
                info.language,

            "language_probability":
                round(
                    info.language_probability,
                    4
                ),

            "duration":
                round(
                    info.duration,
                    2
                ),

            "text":
                " ".join(
                    transcript
                ),

            "segments":
                timestamped_segments
        }


    except subprocess.CalledProcessError:

        raise HTTPException(
            status_code=500,
            detail="FFmpeg failed to extract audio"
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if input_path:

            try:
                os.remove(
                    input_path
                )
            except Exception:
                pass


        if audio_path:

            try:
                os.remove(
                    audio_path
                )
            except Exception:
                pass


# =========================
# RECAP
# =========================

@app.post("/recap")
async def create_recap(
    request: RecapRequest
):

    if groq_client is None:

        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured"
        )


    text = request.text.strip()


    if not text:

        raise HTTPException(
            status_code=400,
            detail="Transcript is empty"
        )


    try:

        response = (
            groq_client
            .chat
            .completions
            .create(

                model=
                    "openai/gpt-oss-120b",

                messages=[

                    {
                        "role": "system",

                        "content": """
You are a professional Myanmar movie recap writer.

The input transcript may be in Chinese, English,
Korean, Japanese, or another foreign language.

Understand the story first and rewrite it as a natural
Myanmar movie recap narration.

Rules:

- Write ONLY in natural Burmese Myanmar language.
- Do NOT output Chinese, English, Korean, Japanese,
  or other foreign languages.
- Do NOT translate word-for-word.
- Keep important story events accurate.
- Do not invent events.
- Keep important character names and details.
- Remove unnecessary repetition.
- Do not use bullet points.
- Write as continuous narration.
- Make it natural and easy to listen to.
"""
                    },

                    {
                        "role": "user",
                        "content": text
                    }

                ],

                temperature=0.3,

                max_tokens=4000
            )
        )


        recap_text = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


        return {

            "success": True,

            "recap":
                recap_text
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================
# SRT TIME FORMAT
# =========================

def format_srt_time(
    seconds
):

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    milliseconds = int(
        round(
            (seconds - int(seconds))
            * 1000
        )
    )


    if milliseconds >= 1000:

        milliseconds = 0
        secs += 1


    if secs >= 60:

        secs = 0
        minutes += 1


    if minutes >= 60:

        minutes = 0
        hours += 1


    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


# =========================
# TTS
# =========================

@app.post("/tts")
async def create_tts(
    request: TTSRequest
):

    text = request.text.strip()


    if not text:

        raise HTTPException(
            status_code=400,
            detail="Recap text is empty"
        )


    voices = {

        "thiha":
            "my-MM-ThihaNeural",

        "nilar":
            "my-MM-NilarNeural"
    }


    selected_voice = voices.get(
        request.voice.lower()
    )


    if not selected_voice:

        raise HTTPException(
            status_code=400,
            detail="Invalid voice. Use thiha or nilar."
        )


    audio_path = None
    srt_path = None


    try:

        audio_file = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            )
        )

        audio_path = audio_file.name

        audio_file.close()


        srt_file = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".srt"
            )
        )

        srt_path = srt_file.name

        srt_file.close()


        communicate = (
            edge_tts.Communicate(
                text,
                selected_voice,
                rate="+0%",
                volume="+0%",
                pitch="+0Hz"
            )
        )


        subtitles = []


        with open(
            audio_path,
            "wb"
        ) as audio:

            async for message in (
                communicate.stream()
            ):

                if (
                    message["type"]
                    == "audio"
                ):

                    audio.write(
                        message["data"]
                    )


                elif (
                    message["type"]
                    == "WordBoundary"
                ):

                    offset = (
                        message["offset"]
                        / 10_000_000
                    )

                    duration = (
                        message["duration"]
                        / 10_000_000
                    )

                    word = (
                        message["text"]
                    )


                    subtitles.append(
                        {
                            "start":
                                offset,

                            "end":
                                offset
                                + duration,

                            "text":
                                word
                        }
                    )


        # =========================
        # CREATE SRT
        # =========================

        with open(
            srt_path,
            "w",
            encoding="utf-8"
        ) as srt:

            for index, item in enumerate(
                subtitles,
                start=1
            ):

                start = (
                    format_srt_time(
                        item["start"]
                    )
                )

                end = (
                    format_srt_time(
                        item["end"]
                    )
                )


                srt.write(

                    f"{index}\n"
                    f"{start} --> {end}\n"
                    f"{item['text']}\n\n"

                )


        # Read files into memory
        # before deleting temp files

        with open(
            audio_path,
            "rb"
        ) as audio:

            audio_data = (
                audio.read()
            )


        with open(
            srt_path,
            "rb"
        ) as srt:

            srt_data = (
                srt.read()
            )


        return {

            "success": True,

            "voice":
                request.voice,

            "audio":
                audio_data.hex(),

            "srt":
                srt_data.decode(
                    "utf-8"
                )
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if audio_path:

            try:
                os.remove(
                    audio_path
                )
            except Exception:
                pass


        if srt_path:

            try:
                os.remove(
                    srt_path
                )
            except Exception:
                pass
