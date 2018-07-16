import wave
import numpy as np

def get_mix_audio(path1, path2):
    wave1_file = wave.open(path1, 'r')
    wave2_file = wave.open(path2, 'r')

    wave1_data = get_frames(wave1_file)
    wave2_data = get_frames(wave2_file)

    print("wave1_data")
    print_wave_info(wave1_file)
    print(wave1_data)
    print("wave2_data")
    print_wave_info(wave2_file)
    print(wave2_data)

    mix_wave_data = (wave1_data / 2 + wave2_data / 2).astype(np.int16)
    save_wav(mix_wave_data, "./data/mixed.wav")

    wave1_file.close()
    wave2_file.close()

    return mix_wave_data, wave1_data, wave2_data

def save_wav(data, path):
    w = wave.Wave_write(path)
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(data)
    # print("mixed_data")
    # print_wave_info(w)
    w.close()


def get_frames(wave_file):
    buf = wave_file.readframes(8000)
    # buf = wave_file.readframes(wave_file.getnframes())

    if wave_file.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype='int16')
    elif wave_file.getsampwidth() == 4:
        data = np.frombuffer(buf, dtype='int32')

    return data

def print_wave_info(wave_file):
    print("  Channel num : ", wave_file.getnchannels())
    print(" Sample width : ", wave_file.getsampwidth())
    print("Sampling rate : ", wave_file.getframerate())
    print("    Frame num : ", wave_file.getnframes())
