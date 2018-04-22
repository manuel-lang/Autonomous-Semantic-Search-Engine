import gensim.downloader as api
import http.server
import math
import pprint
import numpy as np
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from pymongo import MongoClient
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions

os.chdir(os.path.join(os.path.abspath(os.curdir),'out/0/'))

def mongo_connect():
    client = MongoClient('localhost', 27017)
    db = client.stanford_data
    collection = db.document_collection
    documents = collection.documents
    return documents

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()
    def do_GET(s):
        """Respond to a GET request."""

        # Query parameter
        t = s.path.replace('/', '')
        t = t.replace('%20',' ')
        print(t)

        # Entity recognition
        #process entities and keywords
        natural_language_understanding = NaturalLanguageUnderstandingV1(
          username='cccb5076-87bd-4992-b99e-29a0f258460b',
          password='Prop61GOuNtl',
          version='2018-03-16')

        response = natural_language_understanding.analyze(
        text=t,
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
            print("cos-measure")
            prod = np.dot(v1, v2)
            len1 = math.sqrt(np.dot(v1, v1))
            len2 = math.sqrt(np.dot(v2, v2))
            return prod / (len1 * len2)

        def getVectorsOf(model, text):
            print("getVectorsOf")
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
        print("vectorized doc")


        # Aggregate results - NOTE redundant documents
        threshold = 0.2
        cos_values = []
        out_values = []
        for document in schema.find():
            cos_value = cosine_measure(document['word2vec'], query_vector)
            if cos_value > 0.2:
                for entity in response['entities']:
                    if entity['text'] in [x[1] for x in document['entities']]:
                        out_values.append((document, cos_value))
                for keyword in response['keywords']:
                    if keyword in [x[0] for x in document['keywords']] and not (document, cos_value) in out_values:
                        out_values.append((document, cos_value))

        res = sorted(out_values, key=lambda x: x[1])
        print(res)

        # Parse JSON

        # Send response as JSON
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()

        # Write JSON
        s.wfile.write(b"<html><head><title>Title goes here.</title></head>")
        s.wfile.write(b"<body><p>This is a test.</p>")
        # If someone went to "http://something.somewhere.net/foo/bar/",
        # then s.path equals "/foo/bar/".
        s.wfile.write(b"</body></html>")

def run(server_class=HTTPServer):
    server_address = ('', 3000)
    httpd = server_class(server_address, MyHandler)
    httpd.serve_forever()

schema = mongo_connect()
model = api.load("glove-wiki-gigaword-300")  # download the model and return as object ready for use
print("model loaded")
run()
