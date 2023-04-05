from speech_recognition import Recognizer, AudioFile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def analyse():
  recognizer = Recognizer()
  with AudioFile('chile.wav') as audio_file:
    audio = recognizer.record(audio_file)
    text = recognizer.recognize_google(audio)
    if len(text) > 1:
      return text
    else:
      return "Sorry, we did not find any speech in the audio file"

def mooder():
  text = analyse()
  nltk.download('vader_lexicon')
  analyzer = SentimentIntensityAnalyzer()
  mood = analyzer.polarity_scores(text)
  if mood['compound'] > 0:
    return "Positive mood detected"
  else:
    return "Negative mood detected"
  
print(mooder())
