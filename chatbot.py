import numpy as np
import nltk
import string
import random


f=open('bot.txt', 'r',errors = 'ignore')
raw_doc=f.read()
raw_doc=raw_doc.lower() #convrt text into lower case
nltk.download('punkt') #using the punkt tokenizer
nltk.download('wordnet') #using the wordnet dictionary
sent_tokens = nltk.sent_tokenize(raw_doc) #convrt doc to list of sentence
word_tokens = nltk.word_tokenize(raw_doc) #convert doc to list of words

lemmer = nltk.stem.WordNetLemmatizer()
#Wordnet is a semanticaally-oriented dictionary of english included in NLTK
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def lemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# DEFINING THE GREETING FUNCTON

GREET_INPUT = ("hello", "hi", "greetings", "sup", "hey",)
GREET_RESPONSES = ["hi", "hey", "hi there", "hello", " I am glad! You are talking to me"]
def greet(sentence):

    for word in sentence.split():
        if word.lower() in GREET_INPUT:
            return random.choice(GREET_RESPONSES)
        

# RESPONSE GENERATION

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def responce(user_responce):
    robol_responce=''
    TfidfVec = TfidfVectorizer(stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robol_responce=robol_responce+"I am sorry! I don't understand you"
        return robol_responce
    else:
        robol_responce = robol_responce+sent_tokens[idx]
        return robol_responce
    

# DEFINING CONVERSATION START/END PROTOCOL

flag=True
print("BOT: My name is Medic-Bot.let's have a conversation! Also, if you want to exit a tr any time, just say bye!")
while(flag==True):
    user_responce = input()
    user_responce=user_responce.lower()
    if(user_responce!='bye'):
        if(user_responce=='thanks' or user_responce=='thank you' ):
            flag=False
            print("BOT: You are welcome..")
        else:
            if(greet(user_responce)!=None):
                print("BOT: "+greet(user_responce))
            else:
                sent_tokens.append(user_responce)
                word_tokens=word_tokens+nltk.word_tokenize(user_responce)
                final_words=list(set(word_tokens))
                print("BOT: ",end="")
                print(responce(user_responce))
                sent_tokens.remove(user_responce)
    else:
        flag=False
print("BOT: Goodbye! Take care <3 ")
