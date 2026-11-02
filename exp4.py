import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Record speech
r = sr.Recognizer()

with sr.Microphone() as source:
    print("Speak something:")
    audio = r.listen(source)

# Convert speech to text
text = r.recognize_google(audio)

print("\nRecognized Text:", text)

# NLP preprocessing
words = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))

filtered = [w for w in words if w.isalnum() and w not in stop_words]

print("Tokenized Words:", words)
print("After Stop Word Removal:", filtered)