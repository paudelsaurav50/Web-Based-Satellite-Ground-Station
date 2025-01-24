import wave

with open("gqrx_20211006_111850_436253400_8000000_fc_GFSK2.raw", "rb") as inp_f:
    data = inp_f.read()
    with wave.open("sound.wav", "wb") as out_f:
        out_f.setnchannels(2)
        out_f.setsampwidth(2) # number of bytes
        out_f.setframerate(44100)
        out_f.writeframesraw(data)