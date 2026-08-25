import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    audio = r.listen(source)
query = r.recognize_google(audio, language='hi-IN')  # Hindi
print(query)  # "Mitti ka test karo" nu process pannu [web:23]
