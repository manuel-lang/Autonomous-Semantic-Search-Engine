import gensim.downloader as api
import http.server
import math
import json
import numpy as np
import re
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from pymongo import MongoClient
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions
from flask import Flask, jsonify
from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper
app = Flask(__name__)

def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

def mongo_connect():
    client = MongoClient('localhost', 27017)
    db = client.stanford_data
    collection = db.document_collection
    documents = collection.documents
    return documents

@app.route('/<query>')
@crossdomain(origin='*')
def show_user_profile(query):
    t = query.replace('%20',' ')

    # Entity recognition
    #process entities and keywords
    natural_language_understanding = NaturalLanguageUnderstandingV1(
      username='cccb5076-87bd-4992-b99e-29a0f258460b',
      password='Prop61GOuNtl',
      version='2018-03-16')

    response = natural_language_understanding.analyze(
    text=t,
    language='en',
    features=Features(
        entities=EntitiesOptions(
         sentiment=False,
         limit=50),
        keywords=KeywordsOptions(
          sentiment=False,
          emotion=False,
          limit=20)))

    print(response)

    def cosine_measure(v1, v2):
        prod = np.dot(v1, v2)
        len1 = math.sqrt(np.dot(v1, v1))
        len2 = math.sqrt(np.dot(v2, v2))
        return prod / (len1 * len2)

    def getVectorsOf(model, text):
        vectors = []
        for token in wordpunct_tokenize(text):
            try:
                vectors.append(model[token])
            except:
                pass
        return vectors

    def vectorize_document(text):
        return np.array(getVectorsOf(model, text)).mean(axis=0)

    query_vector = vectorize_document(t)


    # Aggregate results - NOTE redundant documents
    threshold = 0.2
    upper_threshold = 0.4
    out_values = []
    for document in schema.find():
        cos_value = cosine_measure(document['word2vec'], query_vector)
        if cos_value > threshold:
            add = False
            extra = 0
            for entity in response['entities']:
                if entity['text'].lower() in [x[1].lower() for x in document['entities']]:
                    add = True
                    extra += 0.1
            for keyword in response['keywords']:
                if keyword["text"].lower() in [x[0].lower() for x in document['keywords']]:
                    add = True
                    extra += 0.1
            if cos_value > upper_threshold:
                add = True
            if (len(document["extracted_image_paths"]) > 0): extra += 0.15
            if(add): out_values.append((document, cos_value + extra))

    res = sorted(out_values, key=lambda x: x[1])
    res = [x for x in res
           if len(x[0]["document_title_pdf"]) > 10
           and len(x[0]["document_title_pdf"]) < 100
           and x[0]["document_title_pdf"][0].isalpha()
           and len(x[0]["document_summary"]) > 10][:10]

    # Parse JSON

    json_raw = {"cards": []}

    for card in res:
        if (card[0]["document_title_pdf"].isupper()):
            title = card[0]["document_title_pdf"].title()
        else:
            title = card[0]["document_title_pdf"]

        json_raw["cards"].append({
            "title": title,
            "type": card[0]["document_type"],
            "url": card[0]["document_url"],
            "filename": card[0]["filename"],
            "tags": [c[0].lower() for c in card[0]["keywords"][:5]],
            "entities": card[0]["entities"][:5],
            "summary": card[0]["document_summary"],
            "thumbnail": re.sub(r'.*/out', 'http://localhost:3002/', card[0]["thumbnail_path"]),
            "images": [re.sub(r'.*/out', 'http://localhost:3002/', c) for c in card[0]["extracted_image_paths"]],
            "date": card[0]["date"],
            "lingvector": card[0]["lingvector"]
        })

    return jsonify(json_raw)

print("connecting to mongodb...")
schema = mongo_connect()
print("downloading word2vec...")
model = api.load("glove-wiki-gigaword-300")
app.run()
