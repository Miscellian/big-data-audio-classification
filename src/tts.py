import pyttsx3


def text_to_mp3(text, file_path, voice='male', rate=150, volume=0.5):
    converter = pyttsx3.init()
    converter.setProperty('rate', rate)
    converter.setProperty('volume', volume)
    voices = converter.getProperty('voices')
    if voice == 'male':
        voice_id = voices[0].id
    else:
        voice_id = voices[1].id

    converter.setProperty('voice', voice_id)
    converter.save_to_file(text, file_path)
    converter.runAndWait()
    
def text_to_speech(text, voice='male', rate=150, volume=0.5):
    converter = pyttsx3.init()
    converter.setProperty('rate', rate)
    converter.setProperty('volume', volume)
    voices = converter.getProperty('voices')
    if voice == 'male':
        voice_id = voices[0].id
    else:
        voice_id = voices[1].id

    converter.setProperty('voice', voice_id)
    converter.say(text)
    converter.runAndWait()