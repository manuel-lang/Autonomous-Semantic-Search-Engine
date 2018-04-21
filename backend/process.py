import gensim.downloader as api
import io
import os

import json
import textract
import pdfminer
import math
import numpy as np

import sys
import PyPDF2
import nltk
from nltk.tokenize import wordpunct_tokenize
from gensim.summarization.summarizer import summarize
from pdfrw import PdfReader
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions

from wand.image import Image

def filter_entities(inp = [("mongodb", 0.71), ("python", 0.75), ("pymongo", 0.93), ("Person", 0.72)], entity_filter = ["Person", "Location", "Organization", "Company"]):
    out = []
    for val in inp:
        if val[0] in entity_filter:
            out.append(val)
    return out

def map_entites_with_pictures(entities):
    # Jan Crawler
    pass

def process_keywords(keywords):
    # text : 'text', relevance : score to (text, score)
    pass

def process_entities(entities):
    mapped_entites = []# type : 'type', text: 'text', relevance: 'relevance' to (type, text relevance)
    filtered_entities = filter_entities(inp = mapped_entites)
    final_entities = map_entites_with_pictures()
    return final_entities

def get_title_from_meta(input_pdf):
    title = PdfReader(input_pdf).Info.Title
    if title != None: title = title.strip("()").strip()
    if title == "": title = None
    return title

def createPDFDoc(fpath):
    fp = open(fpath, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser, password='')
    if not document.is_extractable:
        raise "Not extractable"
    else:
        return document


def createDeviceInterpreter():
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    return device, interpreter


def parse_obj(objs):
    string_fontsize = []
    for obj in objs:
        if isinstance(obj, pdfminer.layout.LTTextBox):
            for o in obj._objs:
                if isinstance(o,pdfminer.layout.LTTextLine):
                    text=o.get_text()
                    if text.strip():
                        old_size = 0
                        string = ""
                        for c in  o._objs:
                            try:
                                string += c._text
                                old_size = c.size
                            except:
                                pass
                            if isinstance(c, pdfminer.layout.LTChar):
                                pass
                        string_fontsize.append({"string":string, "size":old_size})
        elif isinstance(obj, pdfminer.layout.LTFigure):
            parse_obj(obj._objs)
        else:
            pass
    i = 0
    while i < len(string_fontsize):
        if i != 0 and math.floor(string_fontsize[i-1]["size"]) == math.floor(string_fontsize[i]["size"]):
            string_fontsize[i-1]["string"] += string_fontsize[i]["string"]
            del string_fontsize[i]
            i -= 1
        i+=1

    return string_fontsize

def get_title_without_meta(input_pdf):
    document=createPDFDoc(input_pdf)
    device,interpreter=createDeviceInterpreter()
    pages=PDFPage.create_pages(document)
    interpreter.process_page(next(pages))
    layout = device.get_result()
    string_fontsize = parse_obj(layout._objs)
    title_index = max(range(len(string_fontsize)), key=lambda index: string_fontsize[index]['size'])
    title = string_fontsize[title_index]["string"].replace("\n", " ")
    if len(title) <= 75: return title
    return None

def create_pdf_images(input_pdf, output_path):

    # NOTE nach Größe filtern, Mindestgröße ? 100x100?

    pdf = open(input_pdf, "rb").read()
    startmark = b"\xff\xd8"
    startfix = 0
    endmark = b"\xff\xd9"
    endfix = 2
    i = 0

    njpg = 0
    export_paths = []
    while True:
        istream = pdf.find(b"stream", i)
        if istream < 0:
            break
        istart = pdf.find(startmark, istream, istream+20)
        if istart < 0:
            i = istream+20
            continue
        iend = pdf.find(b"endstream", istart)
        if iend < 0:
            raise Exception("Didn't find end of stream!")
        iend = pdf.find(endmark, iend-20)
        if iend < 0:
            raise Exception("Didn't find end of JPG!")

        istart += startfix
        iend += endfix
        print("JPG %d from %d to %d" % (njpg, istart, iend))
        jpg = pdf[istart:iend]
        file_path = os.path.join(output_path, "jpg%d.jpg" % njpg)
        jpgfile = open(file_path, "wb")
        jpgfile.write(jpg)
        jpgfile.close()

        export_path.append(file_path)

        njpg += 1
        i = iend

    return export_paths

def create_thumbnail(src_filename, pagenum = 0, resolution = 72,):
    src_pdf = PyPDF2.PdfFileReader(open(src_filename, "rb"))
    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(src_pdf.getPage(pagenum))
    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)
    img = Image(file = pdf_bytes, resolution = resolution)
    img.convert("png")
    export_path = src_filename + "_thumb.png"
    img.save(filename = export_path)
    return export_path

def getVectorsOf(model, text):
    vectors = []
    for token in wordpunct_tokenize(text):
        try:
            vectors.append(model[token])
        except:
            pass
    return vectors

def vectorize_document(text):
    model = api.load("glove-wiki-gigaword-300")  # download the model and return as object ready for use
    return np.array(getVectorsOf(model, text)).mean(axis=0)

def mongo_connect():
    client = MongoClient('localhost', 27017)
    db = client.stanford_data
    collection = db.document_collection
    documents = collection.documents
    return documents

def mongo_save(document, schema):
    schema.insert_one(document)

def main(entity_limit = 50, keyword_limit = 20):
    schema = mongo_connect()
    for pdffile, index in os.listdir('nextiterationhackathon2018/pdf'):
        document = {}
        document.text = textract.process("nextiterationhackathon2018/pdf/Waechter2018.pdf").decode("utf-8")
        natural_language_understanding = NaturalLanguageUnderstandingV1(
          username='cccb5076-87bd-4992-b99e-29a0f258460b',
          password='Prop61GOuNtl',
          version='2018-03-16')

        response = natural_language_understanding.analyze(
        text=text,
        features=Features(
            entities=EntitiesOptions(
             sentiment=False,
             limit=entity_limit),
            keywords=KeywordsOptions(
              sentiment=False,
              emotion=False,
              limit=keyword_limit)))

        keywords = response['keywords']
        document.keywords = process_keywords(keywords)
        entities = response['entities']
        document.entities = process_entities(entities)
        if get_title_from_meta(pdffile) == None: document.title = get_title_without_meta(pdffile)
        else: document.title = get_title_from_meta(pdffile)
        if title == None: return
        outpath = os.path.join("out", index)
        document.pdf_images_paths = create_pdf_images(pdffile, os.path.join(outpath, "docimages"))
        document.thumbnail_path = create_thumbnail(pdffile, os.path.join(outpath, "thumbnail"))
        document.vector_representation = vectorize_document(text)
        document.summary = summarize(text.replace('\n', ' '), word_count=100, split=False)
        mongo_save(document, schema)

if __name__ == __main__:
    main()
