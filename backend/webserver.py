import gensim.downloader as api
import http.server
import math
import json
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from pymongo import MongoClient
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions

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
                for entity in response['entities']:
                    if entity['text'].lower() in [x[1].lower() for x in document['entities']]:
                        out_values.append((document, cos_value))
                for keyword in response['keywords']:
                    if keyword["text"].lower() in [x[0].lower() for x in document['keywords']] and not (document, cos_value) in out_values:
                        out_values.append((document, cos_value))
                if cos_value > upper_threshold and not (document, cos_value) in out_values:
                    out_values.append((document,cos_value))

        res = sorted(out_values, key=lambda x: x[1])
        res = [x for x in res
               if len(x[0]["document_title_pdf"]) > 10
               and len(x[0]["document_title_pdf"]) < 100
               and x[0]["document_title_pdf"][0].isalpha()
               and len(x[0]["document_summary"]) > 10][:10]

        # Parse JSON

        json_raw = {"cards":[]}

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
                "thumbnail": card[0]["thumbnail_path"],
                "images": card[0]["extracted_image_paths"],
                "date": card[0]["date"],
                "lingvector": card[0]["lingvector"]
            })


        # Send response as JSON
        s.send_response(200)
        s.send_header("Content-type", "text/json")
        s.send_header("Access-Control-Allow-Origin", "*")
        s.end_headers()

        # Write JSON
        s.wfile.write(bytes(json.dumps(json_raw), 'utf-8'))

def run(server_class=HTTPServer):
    server_address = ('', 3000)
    httpd = server_class(server_address, MyHandler)
    httpd.serve_forever()

schema = mongo_connect()
model = api.load("glove-wiki-gigaword-300")  # download the model and return as object ready for use
print("model loaded")
run()
