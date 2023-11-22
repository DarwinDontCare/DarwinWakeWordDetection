import sounddevice as sd
from scipy.io.wavfile import write
import scipy.io.wavfile as wav
import io
from os import walk
import json
import numpy as np
import pyttsx3
import threading

AUDIO_FILE = r"data/wake_word/"
NOT_WAKE_FILE = r"data/not_wake_word/"

def record_and_save_audio(save_path, n_times = 2100):
    confiramtion = input("do you want to record wake word sounds?(y/n) ")
    if confiramtion.lower() == "y":
        start_mark = input("type the index you want to start at: ")
        input("Press Enter to start recording: ")
        for i in range(n_times):
            if i >= int(start_mark):
                fs = 44100
                seconds = 1
                myRecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                sd.wait()
                write(save_path + str(i) + ".wav", fs, myRecording)
                print(f"Currently on: {i + 1} / {n_times}")
                input("Press Enter to start recording: ")

def speak(text):
    text = "itens não completados da sua lista de afazeres: trabalhar um pouco mais a interpretação de texto da AI, começar a fazer um sistema de reconhecimento de emoções para um deep learning, começar a fazer um modelo que consegue gerar texto baseado em um input, melhorar o sistema de busca em sites para a inteligência artificial, fazer um porte para mobile da inteligência artificial."
    text_speech = pyttsx3.init()
    print(f"Darwin: {text} \n")
    text_speech.setProperty('rate', 200)
    voices = text_speech.getProperty('voices')
    text_speech.setProperty('voice', voices[0].id)
    text_speech.say(text=text)
    if text_speech._inLoop:
        text_speech.endLoop()
    text_speech.runAndWait()

def record_background(save_path, n_times = 4800):
    confiramtion = input("do you want to record backgroun sounds?(y/n) ")
    if confiramtion.lower() == "y":
        start_mark = input("type the index you want to start at: ")
        input("Press Enter to start recording background: ")
        """t = threading.Thread(target=speak, args=("",))
        t.start()"""
        for i in range(n_times):
            if i >= int(start_mark):
                fs = 44100
                seconds = 1
                myRecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                sd.wait()
                write(save_path + str(i) + ".wav", fs, myRecording)
                print(f"Currently on: {i + 1} / {n_times}")

record_and_save_audio(AUDIO_FILE)
record_background(NOT_WAKE_FILE)