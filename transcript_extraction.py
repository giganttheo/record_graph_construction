import whisper

model = whisper.load_model("small")

def transcribe(audio):
    return model.transcribe(audio)
