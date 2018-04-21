import pymongo
import datetime
import os
import pprint
from pymongo import MongoClient

def filter_keywords(inp = [("mongodb", 0.71), ("python", 0.75), ("pymongo", 0.93), ("Person", 0.72)], entity_filter = ["Person", "Location", "Organization", "Company"]):
    out = []
    for val in inp:
        if val[0] in entity_filter:
            out.append(val)
    return out

client = MongoClient('localhost', 27017)
db = client.stanford_data
collection = db.document_collection
documents = collection.documents

for file in os.listdir('nextiterationhackathon2018/pdf'):
    document = {"document_title": "My fancy title",
         "document_summary": "My fancy description!",
         "thumbnail_path": "/path/to/thumbnail.png",
         "extracted_image_paths": ["/path/to/img1.png", "/path/to/img2.png", "/path/to/img3.png", "/path/to/img4.png"],
         "document_url": "/path/to/document",
         "document_parent_url": "/path/to/parent",
         "document_type": "paper",
         "filename": "Hyper dyper AI paper",
         "keywords": [("mongodb", 0.71), ("python", 0.75), ("pymongo", 0.93)], # word, score
         "entities": [("Person", "Andrew Ng", 0.93, "/path/to/ng.png"), ("Location", "Cupertino", 0.34, "/path/to/cupertino.png"), ("Organization", "Stanford University", 0.87, "/path/to/stanford.png")], # entity, value, score, image
         "date": datetime.datetime.now()}
    document_id = documents.insert_one(document).inserted_id

filter_keywords()
