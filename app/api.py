# Use this as a helper file...add all the helper methods or heavy-lifting stuff in this disorganized file.
import requests
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer as detokenizer
from nltk import ngrams
import matplotlib.pyplot as plt 
import pandas as pd
import pickle
#from collections import OrderedDict as OD

import gensim #the library for Topic modelling
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as viz #LDA visualization library

import string
from nltk.stem.wordnet import WordNetLemmatizer

import warnings
warnings.simplefilter('ignore')
from itertools import chain


def fetch_speeches():

    # Open JSON file
    f = open('data/speeches.json')

    data = list(json.load(f))

    # Iterating through the json list
    speeches = []
    i = 0
    for i in range(len(data)):

        speeches.append(data[i])

        i += 1

    # Close file
    f.close()

    #print(json.dumps(speeches))

    # Return list of speeches
    return speeches

def cluster_speeches(speeches, num_topics):
    speeches_df = pd.DataFrame(speeches) 
    df = speeches_df.drop(['title', 'date', 'url'], axis=1)
    df['clean_transcript'] = df['transcript'].apply(clean_speech)

    #create speech words dictionary (contains all unique words from all speeches)
    dictionary = corpora.Dictionary(df['clean_transcript'])

    #create bag of words or document term matrix
    bag_of_words = [dictionary.doc2bow(doc) for doc in df['clean_transcript'] ]

    # Instantiate LDA model
    lda = gensim.models.ldamodel.LdaModel

    # Fit LDA model on the dataset
    ldamodel = lda(bag_of_words, num_topics = num_topics, id2word = dictionary, passes=50, minimum_probability = 0)
    topics = ldamodel.print_topics(num_topics = num_topics)

    # Visualize in jupyter notebook for topic name generation by observing words in clusters
    # lda_display = viz.prepare(ldamodel, bag_of_words, dictionary, sort_topics = False, mds='mmds')
    # pyLDAvis.display(lda_display)

    # Assign the topics to the speeches
    lda_corpus = ldamodel[bag_of_words]

    # for doc in lda_corpus:
    #     print(doc)

    scores = list(chain(*[[score for topic_id, score in topic] \
                      for topic in [doc for doc in lda_corpus]]))

    # Calculate threshold to be used to determine which clusters a speech belongs to. Average of probabilities for cluster assignments.
    threshold = sum(scores)/len(scores)

    # Clusters/Topics/Themes (i.e 0,1,2,3...n) consist of lists of IDs of speeches with similar themes. Ids can be used to identify speech/president in speeches_df. 
    clusters = {}
    for cluster_id in range(num_topics):
        clusters[cluster_id] = [j for i,j in zip(lda_corpus,df.index) if i[cluster_id][1] > threshold]
    
    # Return a dict where key is cluster ID and value is list of IDs os speeches that belong in the cluster/theme.
    return clusters


def clean_speech(text):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    stop_free = ' '.join([word for word in text.lower().split() if word not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join([lemma.lemmatize(word) for word in punc_free.split()])

    return normalized.split()



#####################################
# Stuff imported from sharepoint file #
#####################################

# #nltk.download('all')
# pd.set_option('display.max_rows', None)

# stop_words = stopwords.words('english')
# for word in stopwords.words('english'):
#     stop_words.append(word.title())

# # List of ngrams for testing
# ngrams_list = {1:"unigram",2:"bigram",3:"trigram"}


# # Post ngram extraction clean up list
# exclVals = (",",".","?","$","PRSIDENT","'","-","`","’","I")

# # Number of returned ngrams, adjust to get more or less ngrams
# keep_ngram_lvl = 25 

# # Remove stop words from transcript prior to ngram processing
# def remove_stopwords(transcript):
#     tran_tokenized = word_tokenize(transcript)
#     transcript_clean = [word for word in tran_tokenized if not word in stop_words]
#     tran_rtn = detokenizer().detokenize(transcript_clean)
#     return  tran_rtn

# ## Remove N-grams from exclusion list after ngram processing
# ## exclusion list for punctuation and some common words
# def remove_exclude(ngram_list,exclList): 
#     # Compare ngram to exclusion list. 
#     # If ngram contains value in exlcusion list, return true
#     results = []
#     for ngram in ngram_list:
#         if exclude_test(ngram, exclList) == False:
#             results.append(ngram)
#     return results

# def exclude_test(ngram, exclList):
#     response = False
#     for word in exclList:
#         if ngram.count(word.upper()) > 0:
#             response = True
#         if response == True:
#             break
#     return response

# # Get frequency of N-grams in list and return dictionary with N-gram and count
# def freq(lst):
#     d = {}
#     #Group by and count ngrams, put in dictionary
#     for i in lst:
#         if d.get(i) != None:
#             d[i] += 1
#         else:
#             d[i] = 1
#     # Sort dictionary and limit number of ngrams by keep_ngram_lvl variable
#     key_pairs = ((value,key) for (key, value) in d.items())
#     sortList = sorted(key_pairs, reverse = True)[:keep_ngram_lvl]
#     dlist = dict([(value,key) for (key,value) in sortList])
#     return dlist

# def ngram_extraction(transcript, ngram_type):
#     results = []
#     ngrams_list = ngrams(transcript.split(),ngram_type)
#     for grams in ngrams_list:
#         results.append(grams)
#     return results

# def compile_ngrams(data, data_keys, exclVals, pres):
#     for rec in data:
#         if pres == rec[data_keys[3]]:
#             transcript_val = rec[data_keys[4]]
#             transcript_sw = remove_stopwords(transcript_val)
#             temp_res = {"unigram":[],"bigram":[],"trigram":[]}
#             ## Extract ngrams and load in temp results
#             for k,v in ngrams_list.items():
#                 grams = ngram_extraction(transcript_sw, k)
#                 grams_excl = remove_exclude(grams,exclVals )
                
#                 for i in grams_excl:
#                     temp_res[v].append(i)
#     return temp_res

# def compile_ngrams_merged(merged_transcript, ngram_type):
#     ngrams_list_merged = {1:"unigram"}
#     transcript_sw = remove_stopwords(merged_transcript)

#     ## Extract ngrams and load in temp results
#     for k,v in ngrams_list_merged.items():
#         grams = ngram_extraction(transcript_sw, k)
#         grams_excl = remove_exclude(grams,exclVals )
        
#     return grams_excl

# # Load data
# #path ='C:/Users/Aklown/Documents/GA_Tech/Courses/Data_Vis/Project/presidential_speeches.json'
# path ='./data/speeches.json'

# f = open(path,'r')
# data = json.loads(f.read())
# f.close()


# # In[ ]:


# data_keys = list(data[0].keys())

# # array of president names
# president_names = []
# for name in data:
#     if(name['president'] not in president_names):
#         president_names.insert(0,name['president'])

# president_names_short = president_names[:2] #For testing

# ## Set up results dictionary.
# results = {}


# ## loop through predidents names
# for pres in president_names:
#     results[pres] = {"unigram":{},"bigram":{},"trigram":{}}

#     ## loop through data and compile ngrams by president
    
#     temp_res = compile_ngrams(data, data_keys, exclVals, pres)
                    
#         ## Count ngrams and filter 
#     for k,v in ngrams_list.items():
#         tmp = freq(temp_res[v])
#         #print(tmp)
#         for i,u in tmp.items():
#             if i in results[pres][v]:
#                 results[pres][v][i] += int(tmp[i])
#             else:
#                 results[pres][v][i] = int(tmp[i])
#     print("Done : ",pres)

# #Compile all the ngrams and counts, then sort
# bag_of_ngrams = {"unigram":{},"bigram":{},"trigram":{}}
# for pres in results:
#     for k,v in ngrams_list.items():
#         for i,u in results[pres][v].items():
#             if i in bag_of_ngrams[v]:
#                 bag_of_ngrams[v][i] += int(results[pres][v][i])
#             else:
#                 bag_of_ngrams[v][i] = int(results[pres][v][i])

# #sort bag of ngrams
# for k,v in ngrams_list.items():
#     key_pairs = ((value,key) for (key, value) in bag_of_ngrams[v].items())
#     sortBag = sorted(key_pairs, reverse = True)
#     bag_of_ngrams[v] = dict([(str(value),key) for (key,value) in sortBag])
# print("bag of grams complete")

# #Prep results for JSON output. Change tuple keys to string
# for pres in president_names:
#     for k,v in ngrams_list.items():
#         key_pairs = ((value,key) for (key, value) in results[pres][v].items())
#         temp_res = sorted(key_pairs, reverse = True)
#         results[pres][v] = dict([(str(value),key) for (key,value) in temp_res])

# ### Create output files for ngrams by president and bag of ngrams
# with open("data/president_ngrams.json","w") as presoutfile:
#     json.dump(results, presoutfile)

# with open("data/bag_of_ngrams.json","w") as bagoutfile:
#     json.dump(bag_of_ngrams, bagoutfile)
# print("output files created")

# f = open('data/bag_of_ngrams.json', 'r')
# data = json.loads(f.read())
# f.close()

# #Create frequency dictionary from json file
# def freq_dict(ngram):
    
#     key_list = []
#     for i in data[ngram]:
#         key_list.append(i)
        
#     freq_list = []
#     for i in range(len(data[ngram])):
#         freq_list.append(data[ngram][key_list[i]])
        
#     word_list = []
#     for i in range(len(key_list)):
#         word_list.append(key_list[i].strip("'(),'"))
        
#     return  {word_list[i]: freq_list[i] for i in range(len(word_list))}

# uni_freqs = freq_dict('unigram')
# bi_freqs = freq_dict('bigram')
# tri_freqs = freq_dict('trigram')

# uni_exclude_list = ['great', 'sometimes', 'Chief','considerable', 'accordance', 'issued', 'rate', 'moves', 'makes', 'purchase', 'something', 'seen', 'view', '“', '".\'m"', 'pass', 'different', 'gives', 'four','therewith??', 'Whereas', '".n\'t"', '".\'ve"', '".\'s"', 'people', 'United', 'States', 'would', 'may', 'upon', 'one', 'must', 'made', 'every',  'us', 'new', 'years', 'last', '&', 'year', 'shall', '.', 'part', 'going',  'two', 'PRESIDENT:',  'Applause', 'know',  'Q',  'many', 'get', 'THE',  'think', 'subject', 'per', 'man', '#39;', 'never',  'mdash;',  'three', 'like',  'first', 'ever', 'article', 'know', 'best', 'without', 'object',  'That',  'Our', '"I\'m"', 'His', '."', 'could',  'They', 'might', 'make', 'got', 'amount', 'let', 'well', 'way', 'want', 'right', 'cent', 'act', '"In\'t"','lot', 'hundred', 'effect', 'Department',  'amended', 'There',  'today',  'much',  'commissioners', 'call', 'always', 'This', 'Let', 'young', 'within', 'whose', 'still', 'ratified', 'parties', 'life', 'keep', 'attention', 'agreement', '"I\'ve"',  'since', 'regard', 'recovery', 'means', 'hold', 'conscience', 'back', 'also', 'Mr.', 'troops', 'said',  'common',  'among', 'use',  'purpose', 'protocol', 'program', 'problems', 'necessary', 'interest', 'governor', 'Well', 'New', '—', 'question', 'past', 'men', 'large', 'communicated', 'come',  'agencies',  'works',  'representatives',  'miles', 'measures', 'important', 'countries',  'General',  'yet', 'wish', 'whole', 'vessels', 'things', 'section', 'principles', 'principle', 'peoples', 'good',  'general', 'forward', 'administration', 'women', 'terms', 'supply', 'serve', 'say', 'result', 'reciprocal', 'reasonable', 'others', 'organizations', 'objects',  'million', 'less', 'increase', 'difficult', 'development',  'crew', 'contract', 'combination', 'clear', 'balance',  'San', 'wonderful', 'thus', 'supreme', 'refunding', 'questions',  'ought', 'measure',  'lake', 'intend', 'exclusive', 'deposit', 'day', 'circulation', 'agent', 'White', 'Gentlemen', 'Commissioners',  '3', 'voice', 'vessel', 'tried', 'toward', 'together', 'tell',  'route', 'resolution', 'persons', 'passage', 'navigation', 'longer', 'live', 'given', 'forces', 'farmer', 'far', 'fail', 'course', 'committee', 'become', 'around', 'adopted', 'acknowledgment', '"I\'ll"', 'Granada',  'wisdom', 'unlawful', 'traffic', 'team', 'take', 'suppress', 'small', 'several', 'session', 'select', 'room', 'role', 'riotous', 'record', 'put', 'provided', 'provide', 'privilege', 'price', 'precautionary', 'ordered', 'obstructions', 'neither', 'meet', 'life', 'level', 'leave', 'leadership', 'lead', 'intercourse',  'inauguration', 'higher', 'happy', 'go', 'gentlemen', 'experience', 'election', 'doubt', 'deeply', 'decision', 'courses', 'contemplated', 'construction', 'combinations', 'chance', 'cents', 'cargo', 'captured', 'candidate', 'board', 'assemblages', 'appreciate', 'advance', 'across', 'Webb', 'woe', 'wills', 'wholly', 'waste', 'warning', 'vast', 'therefore', 'taking', 'support', 'supplies', 'service', 'seize', 'secret', 'requisite', 'remote', 'received', 'rather', 'provision', 'property', 'private', 'pay', 'organization', 'order', 'opinion', 'offense', 'occasion', 'number', 'neighbors', 'needs', 'mob', 'merely', 'long', 'hereby', 'fellow', 'execution', 'especially', 'employed', 'called', 'assume', 'arbitration', 'younger', 'world,?', 'working', 'witnesses', 'wishes', 'welfare', 'welcome', 'watchful', 'views', 'vicinage', 'verdict', 'various', 'us', 'unlimited', 'ultimate', 'turmoil', 'triers', 'transmit', 'transmission', 'transcendent', 'tranquil']

# bi_exclude_list = ["United', 'States", "&', '#39;", "&', 'mdash;", "Applause', '.", "THE', 'PRESIDENT:", "per', 'cent", "last', 'session",  "fiscal', 'year", "Senate', 'United", "Secretary', 'State", "Federal', 'Government", "Laughter', '.", "year', 'ending", "world', '&", "report', 'Secretary",  "make', 'sure", "years', 'ago", "within', 'limits", "two', 'Governments", "state', 'Union",  "ending', 'June", "amended', 'Senate", "San', 'Francisco", "June', '30", ".)', 'Q",  "three', 'hundred",  "men', 'women",  "every', 'part",    "wish', 'best",  "commissioners', 'United", "Mr.', 'President",  "would', 'like", "want', 'thank", "two', 'countries", "three', 'years", "new', 'administration",  "liberated', 'areas",    "State', 'Maine",    "New', 'Orleans",  "Let', 'us",  "Chief', 'Magistrate",  "6', 'per", "would', 'effect", "world', 'ever", "us', 'tonight",  "third', 'century",    "session', 'Congress",  "per', 'capita", "past', 'four", "one', 'another", "legitimate', 'titles", "last', 'year", "last', 'fiscal", "hundred', 'millions",  "governor', 'Maine", "go', 'far", "free', 'exercise", "first', 'time", "farmers', '&", "far', 'important",  "exchange', 'ratifications", "every', 'American", "eight', 'years", "circulating', 'medium", "charge', '&", "White', 'House", "Thank,', 'Mr.", "Rio', 'Grande", "Q', 'Thank", "Q', 'Mr.", "President-Elect', 'Obama", "PRESIDENT:', 'Yes", "New', 'Granada", "Nations', 'world", "Mexican', 'minister", "6th', 'July", "30th', 'June", "30,', '1868", "3', 'per", '".\'s", \'still', "*', '*", "#39;', 'affaires", "#39;', '&",  "years', 'since", "water', 'courses", "water', 'communication", "wanted,', 'certain", "various', 'duties",   "upon', 'question", "upon', 'part", "upon', 'acts", "tried', 'level",  "treaty', 'United", "toward', 'recovery", "three', 'major", "survey', 'Rio", "state', 'things", "small', 'degree",  "set', 'apart", "select', 'committee", "said,', 'well", "representatives', 'Central", "relations', 'every", "refunding', 'national", "reason', 'believe", "progress', 'made", "power', 'granted", "policy', 'adopted", "placed', 'hands", "past', 'eight", "part', 'United", "orderly', 'government", "order', 'issued", "one', 'hand", "obstructions,', 'combinations", "next', 'month", "new', 'President", "necessary', 'among", "must', 'never", "moves', 'forward", "move', 'forward", "million', 'Americans", "many,', 'many", "man', 'woman", "man', 'affairs", "level', 'best", "laws', 'United", "late', 'war", "know,', 'people", "joint', 'responsibility", "inauguration', 'President", "important', 'pass", "important', 'numbers", "human', 'rights", "hope', 'may", "half', 'century", "great', 'purposes", "good', 'fortune", "going', 'feel", "get', 'done", "four', 'years", "food', 'products", "first', 'President", "fighting', 'men", "existing', 'banks", "executive', 'department", "every', 'one", "decisions', 'made", "days', 'ago", "dare', 'fail", "crew', 'put", "could', 'done", "coasting', 'trade", "cent', 'bonds", "called', 'upon", "call', 'attention", "burn', 'pits", "built', 'greatest", "bonds', 'United", "best', 'way", "believe', 'Federal", "appointed', 'part", "amount', 'public", "among', 'nations", "also', 'want", "action', 'taken", "Vice', 'President", "States', 'may", "Province', 'New", "New', 'Mexico", "New', 'Brunswick", "Mr.', 'Jefferson", "Mount', 'Vernon", "Government', 'must", "Good', 'luck", "Gentlemen', 'House", "General', 'Government",  "Federal', 'public", "Farm', 'Loan",  "Congress', 'tried", "Congress', 'last", "Central', 'South", "Central', 'Empires", "Balkan', 'states",   ".)', 'want", '."\', \'Well', "#39;', 'organizations", "“', 'medical", "young', 'Americans", "would', 'rather", "would', 'cause", "would', 'best", "would', 'become", "would', 'also", "wood', 'pulp", "willingness', 'accept", "whole', 'amount", "whether', 'want", "went', 'politics", "weight', 'measure", "waterway', 'plan", "war', 'rather", "want', 'say", "want', 'conformity", "vision', 'includes", "view', 'presented", "veterans', 'get", "veterans', '&", "using', 'military", "used', 'expended", "upon', 'subject", "upon', 'state", "upon', 'perform", "upon', 'foreign", "upon', 'capital", "upon', 'assumption", "undoubtedly', 'multiplied", "two', 'years", "two', 'peoples",  "true', 'greatness", "troops', 'ordered",  "transaction', 'modern", "told', 'us", "today', 'world", "times', 'greater", "time', 'limit", "time', 'come",  "tie', 'interest", "thus', 'deprived", "three', 'thousand", '"things\'m", \'proudest',  "temporary', 'restraining", "temple', 'liberty",  "taking', 'part", "take', 'steps",  "supply', 'natural",  "strategic', 'balance",  "statute', 'books", "staff,', 'know", "spokesmen', 'Central", "sons', 'Republic", "solid', 'foundation", "small', 'band", "since', 'adoption", "ship', 'canal", "shall', 'asked", "several', 'Balkan", "settlement', 'boundary", "serve', 'citizens", "security', 'circulation", "security', 'bill", "secure', 'access", "section', 'bill",  "say', 'half", "say', 'every", "savings', 'banks", "sailing', 'vessels",  "running', 'boundary", "role', 'space",  "right', 'thing", "right', 'make", "restraining', 'order", "respectively', 'claim",  "reside', 'Great", "reserved', 'rights", "representatives', 'Dominion",  "report', 'state", "report', 'national", "remain', 'first", "relations', 'two", "relations', 'Dominion", "regard', 'ratification", "reduced', 'cost", "recognized', 'legal", "rather', 'let", "rather', 'cut",  "questions', 'arising", "put', 'board", "purchase', 'deposit", "provision', 'made",  "protection', 'contemplated", "private', 'life", "price', 'support", "present', 'session", "practical', 'business", "powers', 'vested", "powers', 'General", "power', 'every", "power', 'construct", "power', 'Congress", "postal', 'savings", "portion', 'territory", "political', 'economic", "placed', 'free", "persons', 'connected",  "peoples', 'world", "people', 'made", "people', 'country", "peculiarly', 'province",  "peace', 'Middle", "passenger', 'crew", "passed', 'since", "passage', 'present", "parts', 'country", "organizations', 'farmers", "order,', 'one", "order', 'facilitate", "order', 'city", "opportunity', 'autonomous", "opening', 'meeting", "open', 'debate", "onward', 'march", "one', 'first", "obtain', 'free", "objects', 'war", "number', 'troops",  "new', 'surveys", "new', 'crew", "new', 'closeness", "new', 'banks", "never', 'seen", "never', 'saw", "never', 'hopeful", "need', 'go", "necessities', 'life", "natural', 'products", "mutual', 'exercise", "mutual', 'exchange", "must', 'work", "must', 'needs", "must', 'lead", "multiplied', 'weakness", "move', 'country", "mouth', 'San", "members', 'White", "measure', 'refunding", "may', 'even", "many', 'forms", "manufacturers', '&", "makes', 'us", "makes', 'America", "make', 'America", "major', 'difficulties", "made', 'free", "made', 'difference", "made', 'city",  "life', 'forever", "lieutenant-governor', 'New", "let', 'us", "let', 'chance", "lemon', 'ranch", "legislative', 'powers",  "latitude', 'construction", "last', 'night", "last', 'Session", "last', '4", "issued', 'United", "international', 'guarantees", "intermediate', 'levels", "interest', 'ceased", "interest', '3", "insure', 'continuance", "information', 'given", "influence', 'United", "increase', 'price",  "hundred', 'years",  "honor', 'Gus",  "held', 'time", "happy', 'occasion",  "greatness', 'comes", "greatest', 'man", "great,', 'great", "great', 'pleasure", "great', 'maritime", "great', 'man", "good', 'government",  "given', 'new", "given', 'Congress", "get', 'sense", "generation', 'must", "general', 'power", "friends', 'allies", "freedom', 'man", "free', 'secure", "form', 'subject", "forces', 'people", "force', 'peace", "force', 'makes", "far', 'better", "fair', 'dealing", "facilitate,', 'promote", "exercise', 'jurisdiction", "executive', 'ability", "execution', 'laws", "exclusive', 'jurisdiction", "everyday', 'citizens", "every', 'citizen", "every', 'case", 'entitled\', \'"act', "end', 'cold",  "difficult', 'times",  "country', '.", "could', 'get", "could', 'become", "continue', 'work", "continue', 'act", "contending', 'parties", "construct', 'roads", "concerning', 'controversies", "communication', 'Atlantic",  "commercial', 'nations",  "combinations,', 'assemblages",  "citizens', 'deeply", "carrying', 'effect", "candidate', 'Governor",  "canal', 'would",  "business', 'man", "business', 'ability", "board', 'passengers", "billions', 'around", "best', 'put", "attention', 'Government", "applauding', 'today", "among', 'several", "always', 'remember", "age', '21", "affairs', 'life", "advice', 'regard", "adopt', 'precautionary", "across', 'isthmus", "across', 'America", "acknowledgment', 'independence", "abundant', 'depth", "York', 'San", "West', 'Indies", "War', 'II", "United', 'States;", "States', 'avoid", "State', 'Governments", "Senate', 'Gentlemen", "San', 'Carlos", "Revolutionary', 'Council", "President', 'United", "President', 'Gorbachev", "Posts', 'within", "Pacific', 'States", "North', 'South", "New', 'York", "Naval', 'force", "National', 'Government", "Mexico', 'Central", "Houses', 'Congress", "Guy', 'Gillette", "Gus', 'Grissom", "Grissom,', 'first", "Granada', 'protection",  "Gentlemen', 'Senate", "General', 'State",  "February,', '1847", "Democratic', 'candidate",  "Commissioners', 'appointed", "Central', 'America", "American', 'nations", "American', 'crew", "America', 'must", "America', 'force", "80th', 'Congress", "400,000', 'men", "4', 'years", "10th', 'February", "10,000', 'miles", ".)', 'Whether", ".)', 'Tonight", '".\'s", \'great', '".\'s", \'always', '".\'m", \'grateful', 'Applause\', ".)\'s"', '"do-nothing"\', \'80th', "’,', 'despite", "—and,', 'would", "—and', 'T.R", "zealous', 'efforts", "zealous', 'cooperation.", "zealous', 'attention", '"you—you\'re", \'going', "younger', 'abler", "young', 'women", "young', 'wife", "young', 'old", "young', 'must", "young', 'men", "young', 'lawyer", "young', 'flower", "young', 'field", "young', 'boy", "young', 'Cubans", "yielding', 'interest", "yielded', 'friendly", "yield', 'temptation", "yield', 'surplus", "yield', 'fullness", "yet,', 'thank", "yet,', 'little", "yet', 'written", "yet', 'sufficiently", "yet', 'successful", "yet', 'spoken", "yet', 'removed", "yet', 'received", "yet', 'realize", "yet', 'principally", "yet', 'hoped", "yet', 'happen", "yet', 'great", "yet', 'fully", "yet', 'fought", "yet', 'far", "yet', 'convinced", "yet', 'apparent", "yet', 'another", "yet', 'House", "yesterday,', 'involved", "years;', 'officers", "years--wish', 'could", "years,', 'wish", "years,', 'public", "years,', 'highest", "years', 'unrequited", "years', 'toil", "years', 'room", "years', 'past", "years', 'office", "years', 'made", "years', 'life", "years', 'kept", "years', 'blessed", "years', 'ahead", 'years\', ".\'s"',  "year,', 'together", "year', 'passes", "year', 'considerable", "year', 'already", "year', '2000", "wrought', 'exposition?", "wrong', 'upon", "written', 'America", "wringing', 'bread", "wounds,', 'care", "would', 'unworthy", "would', 'unquestionably", "would', 'unleashed", "would', 'unbecoming", "would', 'touched", "would', 'thus", "would', 'take", "would', 'subject", "would', 'seem", "would', 'role", "would', 'rend", "would', 'make", "would', 'live", "would', 'left", "would', 'intervene", "would', 'injurious", "would', 'inflict", "would', 'improper", "would', 'expect", "would', 'embrace", "would', 'disastrous", "would', 'direct", "would', 'difficult", "would', 'desire", "would', 'deplorable", "would', 'contrary", "would', 'clinging", "would', 'certainly", "would', 'broken", "would', 'advanced", "would', 'accept", "worthy;', 'rest", "worthy', 'consideration", "worth', 'price", "worse', 'instead",  "world,?', 'whose", "world,', 'give", "world,', 'ever", "world', 'unless",  "world', 'stake", "world', 'single", "world', 'principle", "world', 'possess", "world', 'offenses;", "world', 'look",  "world', 'experience", "world', 'embody",  "world', 'conflicts", "world', 'chance",  "works', 'remain", "workingmen', 'throughout", "working', 'plan", "work,', 'bind", "work', 'thoroughly", "work', 'prepare", "work', 'live", "work', 'go", "work', 'done", 'work\', \'."', "word', 'deed", "wonted', 'caution", "wonderful', 'turnout", "wonderful', 'medium", "wonderful', 'industrial", "women', 'directly", "woe', 'man", "woe', 'due", "wives', 'families", "witnesses', 'observers",  "witnessed', 'four", "without', 'war-seeking", "without', 'war", "without', 'vanity", "without', 'toleration", "without', 'resistance", "without', 'remark", "without', 'reference", "without', 'means", "without', 'looking",  "without', 'harm", "without', 'firing", "without', 'fanfare", "without', 'entering", "without', 'criminal", "without', 'committing", "without', 'carried", "within', 'said", "within', 'energies", "within', 'constitutional", "within', 'confines", "within', 'city",  "within', 'United", "within', 'State",   "wishing', 'best", "wishes', 'happiness", "wished,', 'yet", "wished', 'evacuated", "wish', 'understood", "wish', 'possible", "wisely', 'consistent", "wise', 'true", "wise', 'deliberation", "wise', 'constitutional", "wisdom', 'solitary", "wisdom', 'see", "wisdom', 'magnanimity", "wisdom', 'efficiency", "wisdom', 'avoid", "wills', 'remove", "wills', 'continue", "widow', 'orphan", "whose', 'ox", "whose', 'name", "whose', 'hand", "whose', 'decisions", "whose', 'character", "wholly', 'restrain", "wholly', 'dependent", "whole', 'subject", "whole', 'spirit", "whole', 'progress", "whole', 'population", "whole', 'number", "whole', 'history", "white', 'inhabitants;", "whether', 'would", "whether', 'torn", "whether', 'one", "whether', 'measures", "whether', 'information", "whether', 'Old", "whereof', 'hereunto", "whenever', 'satisfactory", "whatever', 'aspect", "wet', 'landing", "well', 'must", "well', 'military--peril", "well', 'known", "well', 'fortunate", "well', 'expressed", "well', 'every", "welfare', 'prosperity.", "welfare', 'disposition", "welfare', 'Union", "welcome', 'good", "wealth', 'piled", "ways', 'difficult", "way', 'connected", "watchful', 'cares", "waste', 'millions", "waste', 'consequent", "warning', 'persist", "warning', 'especially", "warn', 'persons", "war,', 'disappointment", "war,', 'also", "war', 'speedier", "war', 'southern", "war', 'parties", "war', 'often", "war', 'extermination", "wants', 'assured", "want', 'clearly", "waited', 'precautionary", "votes', 'declared", "votes', 'cast", "votes', 'President", "voted,', 'tie", "vote', 'impossible", "vital,', 'useful", "vital', 'useful", "vital', 'result", "vital', 'part", "vital', 'interest", "violation', 'public", "views', 'advocates", "vicious', 'propensities", "vicinage', '.", "vested', 'law", "verdict', 'rest", "vast', 'importance", "vacillation', 'decisive", "us,', 'sweetener", "us,', 'air", "us', 'yielded", "upright', 'course", "upon', 'table", "upon', 'revolutionary", "upon', 'responsibility", "upon', 'general", "upon', 'disposition", "unlimited,', 'irrevocable", "unless', 'Houses", "unlawfully', 'participating", "ultimate', 'ensuing", "turmoil,', 'bustle", "troops', 'first", "troops', 'city", "troops', 'assembled", "triers', 'vicinage", "tried', 'happiest", "treaty', 'sure", "treaty', 'result", "treaty', 'formulated", "treaty', 'fail", "treaty', 'arbitration", "treaty', 'affords", "treatment', 'guilty", "transpiring', 'around", "transmit', 'herewith", "transmission', 'expression", "transcendent', 'good", "tranquil', 'irresponsible", "traditions,', 'common", "tongue', 'joined",   "thus', 'tried", "therewith??', 'verdict", "therefore,,', 'Grover", "therefore,', 'actually", "theatre', 'public", "testimony', 'whereof", "testimony', 'native", "taken,', 'defrauded?", "sweetener', 'every", "submit:', 'testimony", "stern', 'necessities", "stations', 'called"]

# tri_exclude_list = ["world', '&', '#39;", "year', 'ending', 'June", "ending', 'June', '30",  "Laughter', '.)', 'Q", "fiscal', 'year', 'ending", "last', 'session', 'Congress", "last', 'fiscal', 'year", "commissioners', 'United', 'States", "amended', 'Senate', 'United",  "States', 'Great', 'Britain", "farmers', '&', '#39;", "charge', '&', '#39;", "Thank,', 'Mr.', 'President", "Q', 'Thank,', 'Mr.", "Mexican', 'minister', 'foreign", "June', '30,', '1868",  "6', 'per', 'cent", "&', '#39;', 'affaires", "&', '#39;', '&", "#39;', '&', '#39;", "unlawful', 'obstructions,', 'combinations", "tried', 'level', 'best",  "three', 'hundred', 'seventy-two", "survey', 'Rio', 'Grande", "per', 'cent', 'bonds", "navigation', 'water', 'courses",  "message', '6th', 'July",    "far', 'important', 'numbers", "appointed', 'part', 'United", "United', 'States', 'may", "THE', 'PRESIDENT:', 'Yes",  "Q', 'Mr.', 'President",  "Government', 'United', 'States", "Gentlemen', 'House', 'Representatives:", "Federal', 'public', 'works", "Congress', 'tried', 'level",  "6th', 'July', 'last", "3', 'per', 'cent", "Applause', '.)', 'want", "&', '#39;', 'organizations", "year', 'ending', '30th", "would', 'rather', 'cut",  "whole', 'amount', 'public", "water', 'communication', 'Atlantic", "war', 'rather', 'let", "war', 'Iraq', 'without", "wanted,', 'certain', 'areas;",  "veterans', '&', '#39;",  "undoubtedly', 'multiplied', 'weakness", "unconditional', 'surrender', 'mean", "two', 'Republics', 'island", "tribes', 'within', 'limits", "treaty,', 'amended', 'Senate", "treaty', 'stipulations', 'Mexico", "treaty', 'form', 'ratified", "treaty', 'New', 'Granada", "treaty', 'Mexican', 'Congress", "trade', 'Atlantic', 'Pacific", "time', 'leave', 'office?", "throughout', 'every', 'portion", "three', 'powerful', 'Nations", "third', 'rail', 'American", "third', 'article', 'treaty", "tenth', 'article', 'treaty", "temporary', 'restraining', 'order",  "southern', 'boundary', 'New", "ship', 'canal', 'two", "several', 'Balkan', 'states", "seven', 'hundred', 'millions", "see', 'unity', 'among", "secured', 'free', 'exercise", "say', 'every', 'one", "route', 'Potomac', 'Ohio", "roofs', 'right', 'storm", "roads', 'canals,', 'improve",  "rights', 'citizens', 'United", "restraining', 'order', 'issued",  "rest', 'recreation', 'center", "reside', 'Great', 'Britain", "representatives', 'Dominion', 'Government", "report', 'state', 'Union", "relations', 'two', 'countries",  "ratification', 'Mexican', 'Government", "rate', 'interest', '3", "rail', 'American', 'politics", "pulled', 'roofs', 'right", "public', 'expenditures', 'reached", "provision', 'made', 'law", "proud', 'happy', 'occasion", "protected', 'free', 'enjoyment",  "products', 'necessities', 'life", "port', 'San', 'Juan", "placed', 'free', 'list", "per', 'cent,', 'increase", "per', 'cent', 'reduction", "people', 'pulled', 'roofs", "people', 'United', 'States", "people', '&', '#39;", "peace', 'order', 'city",  "past', 'three', 'years", "past', 'four', 'years", "past', 'eight', 'years", "passenger', 'crew', 'put", "part', 'United', 'States", "organizations', 'farmers', '&", "order', 'facilitate,', 'promote", "opportunity', 'autonomous', 'development", "one', '&', '#39;", "obtain', 'free', 'entry", "obstructions,', 'combinations,', 'assemblages", "ninth', 'article', 'treaty", "new', 'world', 'order",  "negative', 'upon', 'acts", "national', 'security', 'needs", "nation', '&', '#39;", "mutual', 'exercise', 'jurisdiction",   "mouth', 'San', 'Carlos", "ministers', 'France', 'England", "millions,', 'three', 'hundred", "members', 'White', 'House", "measure', 'refunding', 'national", "means', 'placed', 'hands", "may', 'observed,', 'however", "manufacturers', '&', '#39;", "man', '&', '#39;", "made', 'last', 'session", "loan', '$5', 'million", "line', 'New', 'Mexico", "level', 'best', 'put", "late', 'United', 'States", "late', 'Secretary', 'State", "lands', 'within', 'limits", "land', 'agent', 'party", "labor', 'organizations', 'farmers", '"know\'s", \'going\', \'feel', "issued', 'United', 'States", "island', 'St.', 'Domingo", "interest', '3', 'per", "intercourse', 'United', 'States", "information', 'upon', 'state", "information', 'given', 'Congress", "influence', 'United', 'States",  "hundred', 'seventy-two', 'millions", "hundred', 'fifty', 'millions", "humanity', '&', '#39;",   "half', 'century', 'passed", "executive', 'administrative', 'agencies", "execution', 'laws', 'United", "exclusive', 'trust', 'funds", "equal', '9', 'per", "enemy', 'would', 'like", "electric', 'vehicles,', 'creating",  "duty', 'call', 'attention",  "deem', 'duty', 'call", "crew', 'put', 'board", "country', '&', '#39;", "could', 'get', 'sense", "controlling', 'freedom', 'elective", "contract', 'one', 'another", "construct', 'roads', 'canals", "competition', '&', 'mdash;", "communication', 'Atlantic', 'Pacific",  "commercial', 'intercourse', 'United", "commerce', 'among', 'several", "combine', 'contract', 'one", "close', 'last', 'session", "citizens', 'United', 'States", "certain', 'areas;', 'go", "century', 'passed', 'since", "canals,', 'improve', 'navigation", "called', 'upon', 'perform", "built', 'greatest', 'economy", "build', 'electric', 'vehicles", "billions', 'around', 'world", "billion', 'build', 'electric",  "banking', 'association', 'depositing", "association', 'depositing', 'shall", "areas;', 'go', 'far", "answer', 'Secretary', 'State",  "advice', 'regard', 'ratification", "act', '26th', 'May", "York', 'San', 'Francisco",  "Vice', 'President', 'President", "United', 'States', 'avoid", "United', 'States', 'Supreme", "United', 'States', 'America",  "Type', '1', 'diabetes", "Two', 'years', 'ago", "THE', 'PRESIDENT:', 'know", "THE', 'PRESIDENT:', 'hope", "THE', 'PRESIDENT:', 'Well", "THE', 'PRESIDENT:', 'Thank",  "Senate', 'Gentlemen', 'House", "San', 'Francisco', 'next", "San', 'Francisco', 'Conference", "Revised', 'Statutes', 'United", "Province', 'New', 'Brunswick", "President,', 'spoke', 'moment", "New', 'York', 'San", "New', 'Granada', 'protection", "Nazi', 'officers', 'took", "Mr.', 'President,', 'spoke", "Mexico', 'Central', 'South", "Gus', 'Grissom,', 'first", "Government', '&', '#39;", "Gentlemen', 'Senate', 'Gentlemen", "General', 'State', 'Governments", "Francisco', 'next', 'month", "Four', 'years', 'ago",  "Farm', 'Loan', 'Banks", "Europe', '&', '#39;", "Democratic', 'candidate', 'Governor", "Congress', 'last', 'session", "Commissioners', 'appointed', 'part", "70', '33', '32", "10th', 'February,', '1847", "1/2', 'per', 'cent", ".)', 'want', 'thank", '".\'s", \'great\', \'pleasure', "Applause', '.)', 'Whether", "Applause', '.)', 'Tonight", "&', '#39;', 'excise", "#39;', 'excise', 'taxes",  '"do-nothing"\', \'80th\', \'Congress', '"Washington\', \'&\', \'#39;', "”', 'wanted', 'make", "”', 'revitalization', 'American", "”', 'make', 'life", "”', 'initiative', 'people", "”', 'incinerated', 'wastes", "”', 'ground', 'America", "”', 'eight', 'state-of-the-art", "”', 'call', 'building", "”', 'Ukrainian', 'Ambassador", "”', 'Tonight', 'say", "”', 'Another', 'administration", "“', 'medical', 'miracle.", "“', 'medical', 'miracle", "“', 'Made', 'USA.", "“', 'America', 'First", "’,', 'despite', 'frustrations", '—with\', \'"wars\', \'."',  "—and,', 'would', 'like", "—a', 'greater', 'facility",  "—', 'important', 'word", "—', 'great', 'privilege", "—', 'almost', 'everyone", "zones', 'coordinated', 'Berlin", "zone', 'control', 'Germany", "zone', 'Germany—and', 'administration", "zealously', 'contended', 'preservation", "zealous', 'efforts', 'matter", "zealous', 'attention', 'present", '"you—you\'re", \'going\', \'get', "you—particularly', 'new', 'members—as", "you—and,', 'time,', 'people", "youth,', 'continued', 'since", "youth', 'maturity,', 'experience", "youth', 'maintained', 'throughout", "youth', 'Washington', 'showed", "youngest', 'among', 'us", "younger', 'abler', 'minds", "young,', 'smart,', 'fiercely", "young', 'women,', 'like", "young', 'woman', 'writing", "young', 'people', 'country", "young', 'old,', 'rich", "young', 'must', 'know;", "young', 'men', 'women", "young', 'men', 'brought", "young', 'marine,', 'eyes", "young', 'man', 'considered", "young', 'man', 'Old", "young', 'lawyer', 'New", "young', 'flower', 'died", "young', 'field', 'organizer", "young', 'colonel,', 'got", "young', 'boy', 'South", "young', 'Cubans', 'regain", "young', 'Americans,', 'including", "young', 'Americans,', 'gives", "yields', 'protection', 'capitalists", "yielding', 'interest', 'return", "yielded', 'lure', 'shop", "yielded', 'friendly', 'expostulation", "yielded', 'accommodation', 'power", "yield', 'surplus', 'supply", "yield', 'fullness', 'blessings", "yield', 'either', 'principle", "yet,', 'thank', 'God", "yet,', 'somehow,', 'come", "yet,', 'past', 'years", "yet,', 'little', 'success", "yet,', 'continuance', 'name", "yet,', 'always', 'come", 'yet"\', \'—with\', \'"wars', "yet', 'written', 'America", "yet', 'treaty', 'formed", "yet', 'thoroughly', 'removed.It", "yet', 'sufficiently', 'secured", "yet', 'successful', 'result", "yet', 'spoken', 'final", "yet', 'soul', 'subservient", 'yet\', \'realize\', "America\'s"', "yet', 'principally,', 'ignorant", "yet', 'possible', 'announce", "yet', 'known.For', 'full", "yet', 'judiciary', 'forms", "yet', 'hoped', 'efforts", "yet', 'happen,', 'hope", "yet', 'great', 'reason", "yet', 'great', 'American", "yet', 'fully', 'ascertained;", "yet', 'finally', 'concluded", "yet', 'far', 'dominion", "yet', 'entirely', 'removed", "yet', 'convinced', 'justice", "yet', 'communication', 'Commissioners", "yet', 'begun', 'soon", "yet', 'arranged,', 'treaty", "yet', 'apparent', 'treaty", "yet', 'another', 'subject", "yet', 'ages', 'neither", "yet', 'accomplished.It', 'hoped", "yet', 'Soviet', 'Union", "yet', 'House', 'laid", "yesterday,', 'involved', 'Islands--continue", 'yelled,\', \'"Hello,\', \'American', "years—with', 'devotion', 'dedication", "years—I', 'say', 'confidence", "years;', 'officers', 'crews", "years:', 'resurgence', 'national", "years:', 'Space', 'Force", "years--wish', 'could', 'say", "years,', 'without', 'seriously", "years,', 'wish', 'say", "years,', 'two', 'centuries", 'years,\', "soon\'ll", \'time', "years,', 'somewhere,', 'General", "years,', 'righted,', 'order", "years,', 'result', 'decisions", "years,', 'public', 'declarations", "years,', 'one', 'image", "years,', 'men', 'women", "years,', 'government', 'left", "years,', 'forged', 'satisfying", "years,', 'face', 'every", "years,', 'able', 'reverse", "years,', 'American', 'people", "years', 'world', 'again—and", "years', 'work', 'brought", "years', 'still', 'made", "years', 'source', 'true", "years', 'since', 'remark", "years', 'since', 'proclaimed", "years', 'since', 'first", "years', 'since', 'announced", "years', 'since', 'adoption", "years', 'siege', 'Boston", "years', 'seemed', 'bright", "years', 'room', 'witnessed", "years', 'proved,', 'said", "years', 'past', 'midpoint", "years', 'past', 'direction", "years', 'old', 'older", "years', 'office,', 'others", "years', 'national', 'life--century", "years', 'meant', 'mean", "years', 'maintain', 'capabilities", "years', 'made', 'certain", "years', 'life', 'nation", "years', 'later,', 'another", "years', 'large', 'holder", "years', 'kept', 'millions", "years', 'hence', 'children", "years', 'freely', 'elected", 'years\', \'free\', "world\'s"', "years', 'former', 'colony", "years', 'first', 'October", "years', 'find', 'origins", "years', 'ending', 'June", "years', 'effort', 'Washington", "years', 'earned', 'living",  "years', 'develop', 'vaccine", "years', 'constantly', 'absorbing", 'years\', \'come\', \'."', "years', 'ahead,', 'never", "years', 'ago,', 'times", "years', 'ago,', 'still", "years', 'ago,', 'many", "years', 'ago,', 'launched", "years', 'ago,', 'half", "years', 'ago,', 'came", "years', 'ago,', 'another", "years', 'ago,', 'act", "years', 'ago', 'thoughts", "years', 'ago', 'reflects", "years', 'ago', 'proposed", "years', 'ago', 'impenetrable", "years', 'ago', 'freshman", "years', 'ago', 'crime", "years', 'ago', 'Congress", "years', 'ago', '40th", "years', 'ago', '.:", "years', 'age', 'grew", "years', 'Mount', 'Vernon", "years', 'Arizona,', 'seeing", 'years\', ".\'s", \'great', "years', '&', '#39;",  "yearnings', 'millions', 'American", "yearning', 'people', 'exploited", "yearn', 'freedom', 'may", "year,', 'together', 'account", "year,', 'submitted', 'proper", "year,', 'including', 'naval", "year,', 'began', 'actions", "year,', '1910,', '$655,000,000", "year', 'year', 'vital", "year', 'vital', 'matter;", "year', 'open', 'floodgates", "year', 'high', 'devotion", "year', 'continued', 'economy", "year', 'considerable', 'former", "year', 'among,', 'living", "year', 'already', 'caused", "year', '2000', 'much", "year', '11', 'per", "yardstick', 'measure', 'benefits", "wrought,', 'shall', 'also", "wrought', 'tragedy', 'pathetic", "wrought', 'party', 'government", "wrought', 'justice', 'statutes", "wrought', 'exposition?', 'Gentlemen", "wrought', 'destruction', 'healthful", 'wrote:"\', \'led\', \'reflect', 'wrote\', \'letter,\', "couldn\'t"', "wrote', 'describe', 'America", "wrongs', 'done', 'labor", "wronged', 'enfeebled', 'both.It", "wrong,', 'sometimes', 'right", "wrong,', 'shall', 'proper", "wrong', 'upon', 'persons", "wrong', 'represent', 'heroic", "wrong', 'done', 'France", "wrong', 'assertions', 'right", "wrong', 'Administration,', 'top", "written,', 'temptation', 'wrong", "written,', 'many', 'people", "written,', 'know', 'whether", "written,', 'except', 'work", "written,', 'disputes', 'arisen", "written', 'constitution', 'United", "written', 'compact', 'surrenders", "written', 'America', 'knows", "writings', 'made', 'us", "writing', 'speaking,', 'unrestrained", "writing', 'poor', 'bar", "writing', 'opposed', 'incorporation", 'writes,\', \'"Washington\', \'&', "writers', 'upon', 'species", 'writer\', \'"Roman\', \'senate', "write', 'book,', 'probably", "wringing', 'bread', 'sweat",  "wounds,', 'care', 'shall", "wound', 'world', 'war", "would-be', 'troublemakers', 'blueprint", "would,', 'become', 'law", "would', 'wish', 'respond", "would', 'willing', 'conclude", "would', 'upon', 'condition", "would', 'unworthy', 'free", "would', 'unwise', 'authorize", "would', 'unquestionably', 'immediately", "would', 'unleashed', 'every", "would', 'unbecoming', 'representatives", "would', 'truly', 'national", "would', 'touched', 'heart", "would', 'thus', '43", "would', 'take', 'missiles", "would', 'suggest,', 'merely", "would', 'subject', 'suffering", "would', 'still', 'leave", "would', 'steady', 'local", "would', 'somewhat', 'increased", "would', 'single', 'tide", "would', 'single', 'particular", "would', 'set', 'naught", "would', 'seem,', 'grim", "would', 'seem', 'trade", "would', 'seem', 'regular", "would', 'say', 'mother—my", "would', 'role', 'man", "would', 'risk', 'emergence", "would', 'repugnant', 'vital", "would', 'rend', 'Union", "would', 'regarded', 'transaction", "would', 'put', 'upon", "would', 'provide', 'publicity", "would', 'practice', 'limit", "would', 'postpone,', 'defeat", "would', 'postpone', 'effect", "would', 'possible', 'cold", "would', 'permit', 'act", "would', 'necessitate', 'action", "would', 'natural', 'self", "would', 'mutually', 'advantageous", "would', 'materially', 'change",  "would', 'make', 'reduction", "would', 'love', 'talked", "would', 'look', 'anxiety", "would', 'live', 'despair", "would', 'like', 'tell", "would', 'like', 'recompense", "would', 'like', 'go", "would', 'like', 'first", "would', 'like', 'find", "would', 'like', 'ask", "would', 'like', 'acclaim", "would', 'liable', 'attack", "would', 'left', 'without", "would', 'left', 'constructive", "would', 'intervene', 'way", "would', 'injurious', 'inhabitants", "would', 'inflict', 'upon", "would', 'inevitably', 'bring", "would', 'improper', 'dwell",  "would', 'illegal', 'common", "would', 'hopefully', 'approach", "would', 'hazard', 'safety", "would', 'guarantee', 'mistakes", "would', 'great,', 'amounting", "would', 'give', 'would-be", "would', 'give', 'greater",  "would', 'expect', 'accept", "would', 'exhaust', 'process", "would', 'excellent', 'auxiliaries", "would', 'enact', 'contract", "would', 'enable', 'arrest", "would', 'embrace', 'offer", "would', 'effect', 'subjecting", "would', 'effect', 'giving", "would', 'effect', 'excluding", "would', 'effect', 'continuance", "would', 'disastrous', 'demand", "would', 'direct', 'immediate", "would', 'difficult', 'United", "would', 'desired', 'effect", "would', 'desire', 'seize", "would', 'deplorable', 'witness", "would', 'defeat', 'object", "would', 'deem', 'satisfactory", "would', 'contrary', 'traditions", "would', 'contrary', 'established", "would', 'concede,', 'blessing", "would', 'clinging', 'clumsy", "would', 'certainly', 'help", "would', 'certainly', 'afford", "would', 'called', 'sort", "would', 'broken', 'Army", "would', 'best', 'us", "would', 'best', 'applicable", "would', 'benefit', 'sections", "would', 'become', 'subject", "would', 'arise', 'immediately", "would', 'apply', 'equally", "would', 'always', 'get", "would', 'advanced', 'eighteenth", "would', 'admit,', 'carried", "would', 'achieve', 'desired",  "worthy;', 'rest,', 'know", "worthy', 'goals,', 'persuading", "worthy', 'consideration', 'whether", "worth', 'considering', 'whether",  "worse', 'instead', 'better",  "world—everybody', 'wants', 'tell",   "world,?', 'whose', 'ox",  "world,', 'recent', 'increase", "world,', 'one', 'far", "world,', 'give', 'expression", "world,', 'general', 'terms", "world,', 'ever', 'growing", "world,', '&', 'mdash;", '"world\'s", \'people\', \'remember', "world', 'year', '2000", "world', 'within', 'live", "world', 'unless', 'home", "world', 'uniform', 'policy", "world', 'three', 'issues",  "world', 'stake', 'important", "world', 'sort', 'trust", "world', 'single', 'message", "world', 'secured', 'recurrence", "world', 'principle', 'resulting", "world', 'possess', 'necessary",  "world', 'offenses;', 'must", "world', 'obtaining', 'unhampered", "world', 'objects', 'war", "world', 'nearly', 'fifty", "world', 'made', 'fit", "world', 'look', 'America", "world', 'live,', '&", 'world\', \'live\', "knife\'s"', "world', 'live', 'greatest",  "world', 'increase', 'price", "world', 'founded', 'explicitly", "world', 'force', 'selfish", "world', 'food', 'natural", "world', 'experience', 'admonish", "world', 'ever', 'known--(applause)--also", "world', 'embody', 'shared", "world', 'either', 'nourish",  "world', 'connected', 'every", "world', 'conflicts,', 'United", "world', 'coming', 'shores", "world', 'chance', 'work", "world', 'believe', 'supreme", "world', 'among', 'severely", 'world\', \'alive\', ".\'m"', "world', 'advance', 'along", "workshops;', 'slaves,', 'captured", "works', 'remain', 'incomplete", "works', 'accept', 'certain", "workingmen', 'throughout', 'United", "working,', 'keep', 'fighting", "working', 'plan', 'disputes", "working', 'phones', 'late", "working', 'leaders', 'parties", "working', 'late', 'campaign", "workers', 'would', 'rather", "worked', 'way', 'college", "worked', 'office', 'President", "work,', 'indispensable', 'one", "work,', 'great', 'cost", "work,', 'future', 'lies", 'work,\', \'bind\', "nation\'s"', "work,', 'appreciating', 'benefit", "work', 'without', 'delay", "work', 'thoroughly,', 'presenting", "work', 'sections', 'races", "work', 'probably', 'operation", "work', 'prepare', 'Nation", "work', 'possible', 'contingencies", "work', 'magnitude', 'way", "work', 'live', 'raise",  "work', 'great', 'harm", "work', 'done', 'President—every", 'work\', \'."\', \'also', "word', 'deed', 'put",  "wonderful', 'turnout,', 'wonderful", "wonderful', 'tour', 'today", 'wonderful\', \'reception\', ".\'s"', "wonderful', 'medium', 'telegraphy", "wonderful', 'man,', 'Democratic", "wonderful', 'industrial', 'development", "wonderful', 'era', 'made", '"won\'t", \'sympathy\', ".n\'t"', '"won\'t", \'say\', \'much', "women', 'industry,', 'campuses", "women', 'fighting', 'redeem", "women', 'directly', 'engaged", "woe', 'man', 'offense", "woe', 'due', 'offense", "wives', 'families', 'two", "witnesses', 'observers,', 'triers", "witnesses', 'Union', 'emerged", "witnessed', 'horror', 'lingering", "witnessed', 'great', 'moments", "witnessed', 'four', 'major", "witness', 'irregular', 'controversy", "without', 'war-seeking', 'dissolve", "without', 'war,', 'insurgent", "without', 'vanity', 'boastfulness", "without', 'toleration', 'dominant", "without', 'success,', 'directed", "without', 'substantial', 'injury", "without', 'secured', 'mariners", "without', 'risking', 'loss", "without', 'resistance', 'supporters", "without', 'remark', 'information", "without', 'reference', 'particular", "without', 'protecting', 'force", "without', 'parallel', 'history.Fruitful", "without', 'obstructing', 'prohibitory", "without', 'means', 'regaining", "without', 'looking', 'Supreme", "without', 'law,', 'States", "without', 'latitude', 'construction", "without', 'inconvenience;', 'future", "without', 'inadmissible', 'latitude", "without', 'harm', 'industries", "without', 'firing', 'single", "without', 'fanfare,', 'thousands", "without', 'entering', 'treaty", "without', 'delay,', 'provide", "without', 'delay', 'treaty", "without', 'criminal', 'intent", "without', 'complaint', 'burdens", "without', 'compelling', 'withdrawal", "without', 'committing', 'dangerous", "without', 'carried', 'effect", "without', 'authority', 'sanction", "without', 'artificial', 'natural", "without', 'aid', 'strong", "without', 'adequate', 'stock", "without', 'accomplished;', 'Nation", "within,', 'selection', 'Characters", "within', 'thirty', 'days", "within', 'scope', 'agreement", "within', 'said', 'State;", "within', 'purview', 'legislative",  "within', 'power', 'make", "within', 'period', 'withdrawn", "within', 'limits', 'United", "within', 'less', 'hour",  "within', 'jurisdiction;', 'enforce", "within', 'jurisdiction,', 'makes", "within', 'jurisdiction', 'criminal", "within', 'jurisdiction', 'Maine", "within', 'exercised', 'party", "within', 'energies', 'resources", "within', 'constitutional', 'competency", "within', 'confines', '.", "within', 'city', 'State", "within', 'borders', 'may", "within', 'authority', 'limitations", "within', 'United', 'States", "within', 'State', 'Illinois", "withhold', 'signature,', 'cherishing", "withdrawn,', 'provided', 'law", "withdrawal', 'troops', 'therefrom", "withdrawal', 'national-banknotes,', 'thus", "withdrawal', 'currency', 'circulation", "wishing', 'best', 'next", "wishes', 'people', 'sister", "wishes', 'happiness', 'received", "wished,', 'yet', 'great", "wish', 'understood', 'reference", "wish', 'possible', 'two", "wish', 'new', 'President",  "wisely', 'consistent', 'principles", "wise', 'true', 'economy", "wise', 'resolution', 'better", "wise', 'deliberation,', 'assurance", "wise', 'constitutional', 'means", "wisdom', 'solitary', 'efforts", "wisdom', 'see', 'fit", "wisdom', 'patriotism', 'Congress", "wisdom', 'part', 'manufacturers", "wisdom', 'magnanimity,', 'constancy",  "wisdom', 'energy', 'nations", "wisdom', 'efficiency', 'measures", "wisdom', 'avoid', 'causes", "winding', 'banks', 'consequence",  "wills', 'remove,', 'gives", "wills', 'continue', 'wealth", "widow', 'orphan,', 'may", "widening', 'circle', 'intelligence", "widely', 'separated', 'peoples", "whose', 'presence', 'participation", "whose', 'ox', 'taken", "whose', 'name', 'bears", "whose', 'mind', 'ever", "whose', 'hand', 'received", "whose', 'good', 'repeatedly", "whose', 'generous', 'hospitality", "whose', 'decisions', 'stand", "whose', 'character', 'stamped", "wholly', 'restrain', 'vicious", "wholly', 'dependent', 'appropriations", "whole', 'system,', 'taking", "whole', 'subject', 'would", "whole', 'spirit', 'resolution", "whole', 'progress', 'tortuous", "whole', 'population', 'colored", "whole', 'number', 'slaves", "whole', 'matter', 'invite", "whole', 'history', 'shows", "whole', 'history', 'country", "whole', 'country', 'could", "whole', 'attention,', 'absorbs", "whither', 'repaired', 'purpose", "white', 'inhabitants;', 'although", "whether', 'would', 'role", "whether', 'want', 'turn", "whether', 'want', 'go", "whether', 'torn', 'country", "whether', 'one', 'candidates", "whether', 'measures', 'existing", "whether', 'local', 'powers", "whether', 'information', 'conspiracy", "whether', 'additional', 'appropriations", "whether', 'Old', 'New", "wherever', 'perpetrated,', 'many", "whereof', 'hereunto', 'set", "whenever', 'satisfactory', 'information", "whatever', 'considered,', 'manner", "whatever', 'aspect', 'may", "wet', 'landing', 'Atlantic", "well', 'understood', 'favor", "well', 'respect', 'position", "well', 'must', 'show", "well', 'military--peril', 'freedom", "well', 'known', 'public", "well', 'fortunate', 'attempts", "well', 'expressed', 'immediate", "well', 'every', 'interoceanic", "well', 'citizens', 'deeply", "well', 'calculated', 'diffuse", "welfare', 'welfare', 'great", "welfare', 'unconscious),', 'solicitude", 'welfare\', \'great\', "State,\'ll"', "welfare', 'disposition', 'comply", "welcome', 'good', 'give", "week', 'term', 'office", "weave', 'threads', 'coat",  'wealth\', \'piled\', "bondsman\'s"', "weak', 'sister', 'republics", "ways', 'difficult', 'war", 'way,\', "America\'s", \'security', "way', 'connected', 'unlawful", "watered', 'river', 'Aroostook", "watchful', 'cares,', 'labors", "waste', 'millions', 'public", "waste', 'consequent', 'forced", "wars', '20th', 'century", "warranted', 'general', 'forbearance", "warrant', 'belief', 'citizens", "warning', 'persist', 'taking", "warning', 'especially', 'intended", "warned', 'entangling', 'alliances", "warn', 'persons', 'engaged", "war;', 'power', 'nowhere", "war-seeking', 'dissolve', 'Union", "war,', 'previous', 'understanding", "war,', 'one', 'would", "war,', 'may', 'case", "war,', 'insurgent', 'agents", "war,', 'disappointment', 'often", "war,', 'always', 'liable", "war,', 'also', 'form", "war', 'speedier', 'termination", "war', 'southern', 'Republics", "war', 'parties', 'remote", "war', 'often', 'assume", "war', 'extermination', 'white", "war', 'alone', 'declared", "wants', 'assured', 'result", "wanted', 'take', 'town", "want', 'turn', 'clock", "want', 'thank', 'coming", "want', 'go', 'forward", "want', 'clearly', 'understood", "waited,', 'ability', 'new", "waited', 'precautionary', 'measure", "waging', 'war', 'extermination", "vulnerable', 'areas', 'one", "voyage,', 'yet', 'principally", "voyage,', 'employees', 'Brazil", "votes', 'declared', 'standing", "votes', 'cast', 'late", "votes', 'President', 'Vice-President", "voted,', 'tie', 'vote", "vote', 'impossible,', 'must", "voices', 'Asia', 'Latin", "vital', 'result,', 'conditions", "vital', 'part,', 'back", "vital', 'interest', 'peace", "visionary', 'determine', 'real", "vision', 'brought', 'things", "virtue,', 'wisdom', 'magnanimity",  "villages', 'markets--day', 'night--classrooms", "views', 'much', 'reflection", "views', 'merits', 'original", "views', 'expressed', 'message", "views', 'entertained,', 'far", "views', 'advocates', 'immediate", "views', 'Congress', 'make", "view,', 'ground', 'expediency", "view', 'subsequent', 'acquisition", "vicinage', '.,', 'neighbors", "vested', 'law', 'Secretary", "vessels,', 'knowledge,', 'good", "vessels', 'receive', 'whole", "vessel,', 'either', 'directly", "vessel', 'nominally', 'chartered", "vessel', 'delivered,', 'name", "vessel', 'clears', 'United", "verdict', 'rest', 'conscious", "vast', 'tract', 'country", "vast', 'regions', 'occupies", "vast', 'importance', 'whole", "various', 'duties', 'relations", "valuable', 'public', 'servants", "vacillation', 'decisive', 'treatment", "useful', 'lessons', 'us", "use', 'nations', 'equal", "us,', 'think', 'proud", "us,', 'sweetener', 'every", "us,', 'air', 'filled", "us', 'yielded', 'friendly", "us', 'mankind', 'dramatic", "us', 'far', 'beyond", "us', 'conscious', 'crossed", "us', 'accomplishments', 'applauding", "urgency', 'need', 'combat--whether", "urged', 'disinterested', 'friendly", "upright', 'course', 'law", "upon,', 'without', 'entering", "upon', 'yet', 'convinced",  "upon', 'table', 'large", 'upon\', "system\'s", \'survival', "upon', 'revolutionary', 'events", "upon', 'responsibility,', 'attack", "upon', 'projects', 'purposes", "upon', 'persons', 'connected", "upon', 'part', 'portion", "upon', 'objects', 'peculiar", "upon', 'general', 'scheme", "upon', 'disposition', 'disputes", "upon', 'capital', 'public", "upon', 'capital', 'prevent", "upon', 'Government', 'either", "unlimited,', 'irrevocable', 'arbitration", "unless', 'Houses', 'Congress", "unlawfully', 'participating', 'abide", "uniting', 'behind', 'Space", "unemployed,', 'capital', 'idle", "undertaking', 'complete', 'work",   "understood', 'upon', 'disposition", "uncertainty', 'double', 'claim", "unaccomplished,', 'would', 'injurious", "ultimate', 'ensuing', 'benefits", "two', 'receive', 'continue", "two', 'countries', 'reach", "two', 'countries', 'immediately", "two', 'citizens', 'elected", "two', 'Houses', 'order", "turmoil,', 'bustle', 'splendor", "troops', 'ordered', 'city?",  "troops', 'city,', 'kept;", "troops', 'city', 'ought", "troops', 'assembled', 'city", "triers', 'vicinage', '.", "tried', 'happiest', 'auspices", "treaty', 'sure', 'felt", "treaty', 'formulated', 'makes", "treaty', 'fail', 'everywhere", "treaty', 'arbitration', 'matters", "treaty', 'affords,', 'hesitate", "treatment', 'guilty,', 'warning", "transpiring', 'around', 'us",  "transmission', 'expression', 'earnest",  "tranquil', 'irresponsible', 'occupations",  "total', 'amount', '653", "tongue', 'joined', 'together",  "time', 'troops', 'ordered", "time', 'apprehended,', 'followed", "ties', 'common', 'traditions", 'threats\', \'."\', \'circumstances', "thoroughly', 'investigated', 'committee", "therewith??', 'verdict', 'rest", "therefore,,', 'Grover', 'Cleveland", "therefore,', 'actually', 'unlawfully", "theatre', 'public', 'life,;", "testimony', 'whereof', 'hereunto", "testimony', 'native', 'country", "taking', 'part', 'unlawful", "taking', 'part', 'riotous", "taken,', 'defrauded?', 'oppressed", "sweetener', 'every', 'hour", "submit:', 'testimony', 'native", "stern', 'necessities', 'confront", "stations', 'called,', 'obtained", "splendor', 'office,', 'drawn", "society', 'raised,', 'ever", "set', 'hand', 'caused", "seal', 'United', 'States", "said', 'State;', 'Whereas"]

# def exclude(ex_list, diction):
#     for word in ex_list:
#         if word in diction.keys():
#             del diction[word]
#     return diction

# uni_freqs = exclude(uni_exclude_list, uni_freqs)
# bi_freqs = exclude(bi_exclude_list, bi_freqs)
# tri_freqs = exclude(tri_exclude_list, tri_freqs)



