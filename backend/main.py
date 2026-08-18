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


# Free Whisper model
MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")

whisper_model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)


class RecapRequest(BaseModel):
    text: str

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
@app.post("/recap")
async def create_recap(request: RecapRequest):

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

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a professional Myanmar movie recap writer.

Convert the English movie recap transcript
into natural Burmese movie recap narration.

Rules:
- Write ONLY in Burmese.
- Do not include English.
- Keep the original story events accurate.
- Do not invent events.
- Do not translate word-for-word.
- Rewrite naturally like a Myanmar movie recap narrator.
- Make the narration smooth and easy to listen to.
- Keep important story details.
- Remove unnecessary repetition.
- Do not use bullet points.
- Write as continuous narration.
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

        recap_text = (
            response.choices[0]
            .message.content
            .strip()
        )

        return {
            "success": True,
            "recap": recap_text
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.post("/tts")
async def create_tts(request: RecapRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Recap text is empty"
        )

    output_path = None

    try:

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        output_path = output_file.name
        output_file.close()

        communicate = edge_tts.Communicate(
            text,
            "my-MM-ThihaNeural",
            rate="+0%",
            volume="+0%",
            pitch="+0Hz"
        )

        await communicate.save(output_path)

        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename="myanmar-recap.mp3"
        )

    except Exception as e:

        if output_path:

            try:
                os.remove(output_path)
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
