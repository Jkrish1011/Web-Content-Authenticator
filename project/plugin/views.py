from django.shortcuts import render,redirect
from django.http import HttpResponse	
from django.views.generic import View	
from django.template import Context, loader
import json
import csv
from django.db.models import Q
import sqlite3
import os
from pathlib import Path
from django.views.decorators.csrf import csrf_exempt
from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import urllib.request

from bs4.element import Comment
from statistics import mode
import nltk
import random

from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
import codecs   

def convert_to_tokens(input_url):
	tokens_red_slash=str(input_url.encode('utf-8')).split('/')
	all_tokens=[]
	for i in tokens_red_slash:
		tokens=str(i).split('-')
		tokens_dot=[]
		for j in range(0,len(tokens)):
			temp=str(tokens[j]).split('.')
			tokens_dot=tokens_dot+temp
		all_tokens=all_tokens+tokens+tokens_dot
	all_tokens=list(set(all_tokens))
	if 'com'in all_tokens:
		all_tokens.remove('com')
	return all_tokens
 

###############################################


####################################. run
def Visible(element):
  if element.parent.name in ['style' , 'script' ,'[document]' , 'head' ,'title']:
    return False
  if isinstance(element,Comment):
    return False

  return True

def text_from_html(body):
  soup=BeautifulSoup(body,'html.parser')
  texts=soup.findAll(text=True)
  result=filter(Visible,texts)
  return u" ".join(t.strip() for t in result)
  
####################################. run



## it is a function for calculating votes so as to check which classifier we are using 
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)




def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

feature_sets_temp = open("/Users/jk/Documents/Major/plugin_Test/features.pickle", "rb")
feature_sets = pickle.load(feature_sets_temp)
feature_sets_temp.close()



random.shuffle(feature_sets)


training_set = feature_sets[:10000]
testing_set = feature_sets[10000:]


open_file = open("/Users/jk/Documents/Major/plugin_Test/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()



voted_classifier = VoteClassifier(classifier)
                                 

def senti(text_doc):
    feats = find_features(text_doc)
    return voted_classifier.classify(feats)





###############################################

@csrf_exempt
def func_hello(request):
	''' This function is just for testing purposes '''
	try:
		resp = str(request.body,encoding='utf=8')
		resp1 = json.loads((resp))
		return HttpResponse(json.dumps(resp1))

	except Exception as e:
		return HttpResponse(e)


@csrf_exempt
def plugin(request):
	''' This function is for the actual plugin.'''
	try:
		##implementation of codecs 
		with codecs.open("/Users/jk/Documents/Major/plugin_Test/positive.txt", "r", "latin-1") as inputfile:
		    short_pos=inputfile.read()
		with codecs.open("/Users/jk/Documents/Major/plugin_Test/negative.txt", "r", "latin-1") as inputfile:
		    short_neg=inputfile.read()

		all_words = []
		documents = []


		#  here j is adjective , r is adverb, and v is verb but we are removing all the adverb and verb here
		#allowed_word_types = ["J","R","V"]
		allowed_word_types = ["J"]

		for p in short_pos.split('\n'):
		    documents.append( (p, "0") )
		    words = word_tokenize(p)
		    pos = nltk.pos_tag(words)
		    for w in pos:
		        if w[1][0] in allowed_word_types:
		            all_words.append(w[0].lower())

		    
		for p in short_neg.split('\n'):
		    documents.append( (p, "1") )
		    words = word_tokenize(p)
		    pos = nltk.pos_tag(words)
		    for w in pos:
		        if w[1][0] in allowed_word_types:
		            all_words.append(w[0].lower())



		documents_temp = open("/Users/jk/Documents/Major/plugin_Test/documents.pickle", "rb")
		documents = pickle.load(documents_temp)
		documents_temp.close()



		word_features_temp = open("/Users/jk/Documents/Major/plugin_Test/word_features5k.pickle", "rb")
		word_features = pickle.load(word_features_temp)
		word_features_temp.close()


	except Exception as e:
		return HttpResponse(e)
	try:
		resp1 = json.loads(str(request.body,encoding='utf=8'))
		resp = []
		for i in resp1.values():
			resp.append(i)
		
		p = re.compile(r"https")
		exx = {}
		val = []
		for i in range(len(resp)):
		    rs = p.match(resp[i])
		    if(rs == None):
		        val.append(0)
		    else:
		        val.append(1)


		domain1 = ["org", "gov", "mil", "edu", "int"]
		#domain2 = ["ac","ad","ae","in","at","au","bb","us","cl","ch","ca","br","be"]
		domain2 = ["ac","ad","ae","af","ag","ai","al","am","an","ao","aq","ar","as","at","au","aw","ax","az","ba","bb","bd","be","bf","bg","bh", 
		"bi","bj","bl","bm","bn","bo","bq","br","bs","bt","bv","bw","by","bz""ca","cc","cd","cf","cg","ch","ci","ck","cl","cm","cn","co","cr",
		"cu","cv","cw","cx","cy","cz","de","dj","dk","dm","do","dz","ec","ee","eg","eh","er","es","et","eu","fi","fj","fk","fm","fo","fr","ga",
		"gb","gd","ge","gf","gg","gh","gi","gl","gm","gn","gp","gq","gr","gs","gt","gu","gw","gy","hk","hm","hn","hr","ht","hu","id","ie","il","im",
		"in","io","iq","ir","is","it","je","jm","jo","jp","ke","kg","kh","ki","km","kn","kp","kr","kw","ky","kz","la","lb","lc","li","lk","lr","ls",
		"lt","lu","lv","ly","ma","mc","md","me","mf","mg","mh","mk","ml","mm","mn","mo","mp","mq","mr","ms","mt","mu","mv","mw","mx","my","mz","na",
		"nc","ne","nf","ng","ni","nl","no","np","nr","nu","nz","om","pa","pe","pf","pg","ph","pk","pl","pm","pn","pr","ps","pt","pw","py","qa","re",
		"ro","rs","ru","rw","sa","sb","sc","sd","se","sg","sh","si","sj","sk","sl","sm","sn","so","sr","st","su","sv","sx","sy","sz","tc","td","tf",
		"tg","th","tj","tk","tl","tm","tn","to","tp","tr","tt","tv","tw","tz","ua","ug","uk","us","uy","uz","va","vc","ve","vg","vi","vn","vu","wf",
		"ws","ye","yt","za","zm","zw"]

		domain3 = ["com","co","net"]
		val2 = []
		for i in range(len(resp)):
		    splitLink = resp[i].split(".")
		    try:
		    	splink = splitLink[2].split("/")

		    except Exception as e:
		    	splink = splitLink[1].split("/")

		    if splink[0] in domain1:
		        val2.append(3)
		    elif splink[0] in domain3:
		        val2.append(1)
		    elif splink[0] in domain2:
		        val2.append(2)
		    else:
		    	val2.append(0)

		# exx["val"] = val
		# exx["val2"] = val2
		# exx["links"] = resp1

		sentimnt = []
		for i in range(len(resp)):
			try:
				html=urllib.request.urlopen(resp[i])
				abcd = text_from_html(html)
				sentimnt.append(eval(senti(abcd)))

			except Exception as e:
				sentimnt.append(0)

		###############################################
		token = open("/Users/jk/Documents/Major/plugin_Test/tokens.pickle", "rb")
		all_token = pickle.load(token)
		token.close()
		##pickling is used

		load_y = open("/Users/jk/Documents/Major/plugin_Test/y.pickle", "rb")
		y = pickle.load(load_y)
		load_y.close()


		load_corpus = open("/Users/jk/Documents/Major/plugin_Test/corpus.pickle", "rb")
		corpus = pickle.load(load_corpus)
		load_corpus.close()

		vect_url= TfidfVectorizer(all_token)

		x=vect_url.fit_transform(corpus)

		x_train ,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
		filename = '/Users/jk/Documents/Major/plugin_Test/finalized_model.sav'

		value = pickle.load(open(filename, 'rb'))

		value.fit(x_train,y_train)
	


		x_predict=vect_url.transform(resp)
		y_predict=value.predict(x_predict)
		str1 = ''.join(map(str,(y_predict)))
		malicious = list()
		for i in range(len(str1)):
			malicious.append(int(str1[i]))



##########################################################

		ll = len(resp)
		cc = 0
		ss = [[0 for i in range(7)] for j in range(ll)]
		for i in range(ll):
			if(val[i] == 1):
				ss[i][0] = str(0)
				ss[i][1] = str(1)
			else:
				ss[i][0] = str(1)
				ss[i][1] = str(0)

			if(val2[i] == 3):
				ss[i][2] = str(1)
				ss[i][3] = str(0)
				ss[i][4] = str(0)
			elif(val2[i] == 2):
				ss[i][2] = str(0)
				ss[i][3] = str(1)
				ss[i][4] = str(0)
			elif(val2[i] == 1):
				ss[i][2] = str(0)
				ss[i][3] = str(0)
				ss[i][4] = str(1)

			
			if(sentimnt[i] == 0):
				ss[i][5] = str(0)
			else:
				ss[i][5] = str(1)

			if(malicious[i] == 0):
				ss[i][6] = str(0)
			else:
				ss[i][6] = str(1)

		head = ['Http','Https','High','Med','Low','Sentimental','Malicious']



		with open('/Users/jk/Documents/Major/plugin_Test/sample.csv', "w") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(head)
			for i in range(ll):
				writer.writerow((ss[i]))

		df = pd.read_csv('/Users/jk/Documents/Major/plugin_Test/sample.csv')

		cor_df = df.corr()

		score = np.zeros(df.shape[0])

		sp = {}
		f=0
		for i in range(0,df.shape[0]):
			if df.iloc[i,0] == 1: 
				score[i] = np.sum(df.iloc[i,:]*cor_df.iloc[0,:]) -1.0
				sp[f] = (np.sum(df.iloc[i,:]*cor_df.iloc[0,:]) -1.0)
			else:
				score[i] = np.sum(df.iloc[i,:]*cor_df.iloc[1,:]) - 1.0
				sp[f] = (np.sum(np.sum(df.iloc[i,:]*cor_df.iloc[1,:]) - 1.0))
			f+=1

		df['Score'] = score
		df.to_csv('/Users/jk/Documents/Major/plugin_Test/Score.csv')

		return HttpResponse(json.dumps(sp))


	except Exception as e:
		return HttpResponse(e)