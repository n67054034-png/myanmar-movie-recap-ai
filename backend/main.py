import os
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import imageio_ffmpeg
from groq import Groq

app = FastAPI(title="Myanmar Movie Recap AI")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Free Whisper model
MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")

whisper_model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


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

        # Save uploaded video
        suffix = os.path.splitext(
            file.filename
        )[1] or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(await file.read())
            input_path = temp.name


        # Get FFmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


        # Temporary WAV file
        audio_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        audio_path = audio_temp.name
        audio_temp.close()


        # Extract audio from video
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


        # Free Whisper transcription
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True
        )


        transcript = []
        timestamped_segments = []


        for segment in segments:

            text = segment.text.strip()

            if text:

                transcript.append(text)

                timestamped_segments.append({
                    "start": round(
                        segment.start,
                        2
                    ),
                    "end": round(
                        segment.end,
                        2
                    ),
                    "text": text
                })


        return {
            "success": True,
            "filename": file.filename,
            "language": info.language,
            "language_probability": round(
                info.language_probability,
                4
            ),
            "duration": round(
                info.duration,
                2
            ),
            "text": " ".join(transcript),
            "segments": timestamped_segments
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
                os.remove(input_path)
            except Exception:
                pass


        if audio_path:

            try:
                os.remove(audio_path)
            except Exception:
                pass
