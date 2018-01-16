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

		exx["val"] = val
		exx["val2"] = val2
		exx["links"] = resp1

		ll = len(val)
		cc = 0
		ss = [[0 for i in range(5)] for j in range(ll)]
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

		head = ['Http','Https','High','Med','Low']

		with open('/Users/jk/Documents/7th Sem/Major/plugin_Test/sample.csv', "w") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(head)
			for i in range(ll):
				writer.writerow((ss[i]))

		df = pd.read_csv('/Users/jk/Documents/7th Sem/Major/plugin_Test/sample.csv')

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
		df.to_csv('/Users/jk/Documents/7th Sem/Major/plugin_Test/Score.csv')

		return HttpResponse(json.dumps(sp))


	except Exception as e:
		return HttpResponse(e)