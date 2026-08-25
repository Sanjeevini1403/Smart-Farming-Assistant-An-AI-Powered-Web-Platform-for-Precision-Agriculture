from gtts import gTTS
import pygame

def speak_tamil(text):
    tts = gTTS(text=text, lang='ta')  # Tamil
    tts.save('response.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load('response.mp3')
    pygame.mixer.music.play()
