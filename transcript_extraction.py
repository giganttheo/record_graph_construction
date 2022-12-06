import whisper
import warnings

model = whisper.load_model("small")

def transcribe(audio):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.transcribe(audio)
