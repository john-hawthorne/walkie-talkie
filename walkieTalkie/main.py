from pathlib import Path
from openai import OpenAI
from playsound import playsound
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import cutlet
from fuzzywuzzy import fuzz
import os

api_key = os.getenv('OPENAI_API_KEY')


def capture(tts):
    client = OpenAI(
        api_key=api_key
    )
    speech_file_path = str(Path(__file__).parent) + "/question.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=tts
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path


def play_audio(file_path):
    playsound(file_path)


def record_audio(duration=5, sample_rate=44100):
    speech_file_path = str(Path(__file__).parent) + "/response.wav"
    print(f"Recording {duration} seconds of audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Save the recorded audio to a file
    write(speech_file_path, sample_rate, audio_data)
    return speech_file_path


def speech_to_text(file_path, local_iso):
    client = OpenAI(
        api_key=api_key
    )

    audio_file=open(file_path, "rb")
    transcript=client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language=local_iso
    )
    return transcript


def convert_to_romaji(text):
    katsu = cutlet.Cutlet()
    katsu.use_foreign_spelling = False
    return katsu.romaji(text)


def get_percent_diff(expected, actual):
    return fuzz.ratio(expected, actual)


language = input("Choose Language: Spanish (1), French (2), Japanese (3), or English (4) ")
iso = ""
if language == "1":
    iso = "es"
elif language == "2":
    iso = "fr"
elif language == "3":
    iso = "ja"
elif language == "4":
    iso = "en"

# prompt user for question
question = input("What question would you like to ask?")
print(question)

expected_response = input("What is the expected response?")
print(expected_response)

submit = input("Submit? (Y)")

if submit == 'Y':
    path_to_question_file = capture(question)
    play_audio(path_to_question_file)
    path_to_response_file = record_audio()
    actual_response = speech_to_text(path_to_response_file, iso)
    print("Question: " + question)
    print("Expected: " + expected_response)
    print("Actual:" + actual_response.text)

    if language == "3":
        expected_response_romaji = convert_to_romaji(expected_response)
        print("Expected (Romaji):" + expected_response_romaji)
        actual_response_romaji = convert_to_romaji(actual_response.text)
        print("Actual (Romaji):" + actual_response_romaji)
        print(get_percent_diff(expected_response_romaji, actual_response_romaji))
    else:
        print(get_percent_diff(expected_response, actual_response.text))
