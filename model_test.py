import librosa
import numpy as np
from tensorflow.keras.models import load_model
import time
from tensorflow.python import keras
import sounddevice as sd
import tensorflow
import threading
from scipy.io.wavfile import write
import pyaudio
import python_speech_features

fs = 22050
seconds = 1
step = 0.05
M = 50
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 8000
num_mfcc = 16
len_mfcc = 26

tensorflow.compat.v1.disable_eager_execution()

class Wake_Word_listener:
    queue = []
    keras.backend.clear_session()
    pya = pyaudio.PyAudio()
    model = load_model("saved_model/WakeWordModel.pb")
    cancel_wake = False

    def listener(self):
        self.wake = False
        self.prediction_M()
        return self.wake

    def sound_capture(self):
        try:
            self.cancel_wake = False
            stream = self.pya.open(fs, 1, pyaudio.paFloat32, input=True, frames_per_buffer=fs * seconds)
            print('start talking')
            while self.wake == False:
                try:
                    #sys_s = threading.Thread(target=self.Record_sys_sound)
                    #sys_s.start()
                    audio = stream.read(fs * seconds, exception_on_overflow=False)
                    audio = np.frombuffer(audio, dtype=np.float32)
                    self.queue.append(audio)
                    time.sleep(0.01)
                    #blockLinearRms= np.sqrt(np.mean(detect_volume**2))
                    #blockLogRms = 20 * math.log10(blockLinearRms)
                    if self.wake == True:
                        break
                except Exception as e:
                    print(e)
            stream.stop_stream()
            stream.close()
            self.pya.terminate()
            return
        except:
            self.cancel_wake = True
            return


    def prediction_M(self):
        capt_audio = threading.Thread(target=self.sound_capture, daemon=True)
        capt_audio.start()
        while True:
            if len(self.queue) > 2:
                deff = len(self.queue) - 15
                for i in range(deff):
                    self.queue.pop(0)
            if len(self.queue) > 0:
                mfcc_processed = python_speech_features.base.mfcc(self.queue[len(self.queue) - 1].ravel(), 
                                            samplerate=fs,
                                            winlen=0.256,
                                            winstep=step,
                                            numcep=num_mfcc,
                                            nfilt=26,
                                            nfft=5645,
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=False,
                                            winfunc=np.hanning).transpose()

                if mfcc_processed.shape[1] != len_mfcc:
                    diff = len_mfcc - mfcc_processed.shape[1]
                    zeros_to_add = np.zeros((16, diff))
                    mfcc_processed = np.concatenate([mfcc_processed, zeros_to_add], axis=1)

                mfcc_processed = np.float32(mfcc_processed.reshape(1, mfcc_processed.shape[0], mfcc_processed.shape[1], 1))

                predict = self.model.predict(mfcc_processed)
                print(predict)
                if predict[0] < 1.1881564e-11:
                    print("wake word detected!")
                if self.cancel_wake:
                    self.queue.clear()
                    return
                time.sleep(1)

if (__name__ == "__main__"):
    print(8.154866e-28 < 0)
    k = Wake_Word_listener()
    k.listener()