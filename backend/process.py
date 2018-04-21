import datetime
import gensim.downloader as api
import io
import json
import math
import nltk
import numpy as np
import os
import pandas as pd
import pdfminer
import pickle
import PyPDF2
import re
import sys
import pdfrw
import textract
import tldextract
import traceback

from collections import Counter
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from gensim.summarization.summarizer import summarize
from os.path import basename
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfrw import PdfReader
from PIL import Image
from pymongo import MongoClient
from sklearn.base import BaseEstimator
from wand.image import Image
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions

def words(text): return re.findall(r'\w+', text.lower())

class LinguisticVectorizer(BaseEstimator):

    def get_feature_names(self):
        return np.array(
            ['text_length',
             'number_of_paragraphs',
             'average_sent_length',
             'average_word_length',
             'number_of_nouns',
             'number_of_adjectives',
             'number_of_verbs',
             'type_token_relation',
             'hapaxes_index',
             'action_index',
             'number_of_question_marks',
             'number_of_exclamations',
             'number_of_percentages',
             'number_of_currency_symbols',
             'number_of_paragraph_symbols',
             'content_fraction',
             'number_of_cappsed_words',
             'number_of_first_person_pronouns']
        )

    def fit(self, documents, y=None):
        return self

    def __filter(self, string):
        return [w for w in word_tokenize(string) if w.isalpha()]

    def _get_text_length(self, string):
        tokens = self.__filter(string)
        return len(tokens)

    def _get_number_of_paragraphs(self, string):
        return round(string.count('\n') / 2)

    def _get_average_sent_length(self, string):
        tokens = self.__filter(string)
        if len(sent_tokenize(string)) is 0:
            return len(tokens)
        return len(tokens) / len(sent_tokenize(string))

    def _get_average_word_length(self, string):
        tokens = self.__filter(string)
        word_length_list = []
        for word in tokens:
            word_length_list.append(len(word))
        return np.average(word_length_list)

    def _get_number_of_pos(self, string):
        nouns = 0
        verbs = 0
        adj = 0
        for a in pos_tag(self.__filter(string)):
            if a[1] in ['NN', 'NNS', 'NNP', 'NNPS']: nouns += 1
            if a[1] in ['JJ', 'JJR', 'JJS']: adj += 1
            if a[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: verbs += 1
        length = self._get_text_length(string)
        return (nouns/length, verbs/length, adj/length, verbs / (adj + verbs))
        # n, v, a, naq

    def _get_ttr(self, string):
        tokens = self.__filter(string)
        if len(tokens) is 0:
            return 0
        return len(set(tokens)) / len(tokens)

    def _get_hl(self, string):
        words = self.__filter(string)
        fdist = nltk.FreqDist(words)
        hapaxes = fdist.hapaxes()
        if len(words) is 0:
            return len(hapaxes)
        return len(hapaxes) / len(words)

    def _get_number_of_currency_symbols(self, string):
        currencies = ["£","€","$","¥","¢","₩"]
        sum = 0
        for currency in currencies:
            sum += self._get_number_of_symbol(string, currency)
        return sum / self._get_text_length(string)

    def _get_number_of_symbol(self, string, symbol):
        return string.count(symbol) / self._get_text_length(string)

    def _get_content_fraction(self, string):
        tokens = self.__filter(string)
        content = [w for w in tokens if w.lower() not in stopwords.words('english')]
        if len(tokens) is 0:
            return 0
        return len(content) / len(tokens)

    def _get_number_of_cappsed_words(self, string):
        tokens = self.__filter(string)
        return np.sum([t.isupper() for t in tokens if len(t) > 2]) / self._get_text_length(string)

    def _get_number_of_first_person_pronouns(self, string):
        tokens = word_tokenize(string)
        pronouns = ["i","me","my", "mine", "myself","we", "our", "ours", "ourself"]
        sum = 0
        mode = 0
        for word in tokens:
            if word == "``":
                mode = mode + 1
            elif word == "''":
                mode = mode - 1

            if mode <= 0 and word.lower() in '\t'.join(pronouns):
                sum += 1
        return sum / len(tokens)

    def transform(self, documents):
        text_length = [self._get_text_length(d) for d in documents]
        number_of_paragraphs = [self._get_number_of_paragraphs(d) for d in documents]
        average_length_of_sent = [self._get_average_sent_length(d) for d in documents]
        average_word_length = [self._get_average_word_length(d) for d in documents]
        number_of_nouns, number_of_verbs, number_of_adjectives, action_index = self._get_number_of_pos(documents[0])
        type_token_relation = [self._get_ttr(d) for d in documents]
        hapaxes_index = [self._get_hl(d) for d in documents]
        number_of_question_marks = [self._get_number_of_symbol(d, "?") for d in documents]
        number_of_exclamations = [self._get_number_of_symbol(d, "!") for d in documents]
        number_of_percentages = [self._get_number_of_symbol(d, "%") for d in documents]
        number_of_currency_symbols = [self._get_number_of_currency_symbols(d) for d in documents]
        number_of_paragraph_symbols = [self._get_number_of_symbol(d, "§") for d in documents]
        content_fraction = [self._get_content_fraction(d) for d in documents]
        number_of_cappsed_words = [self._get_number_of_cappsed_words(d) for d in documents]
        number_of_first_person_pronouns = [self._get_number_of_first_person_pronouns(d) for d in documents]

        result = np.array(
            [text_length,
             number_of_paragraphs,
             average_length_of_sent,
             average_word_length,
             [number_of_nouns],
             [number_of_adjectives],
             [number_of_verbs],
             type_token_relation,
             hapaxes_index,
             [action_index],
             number_of_question_marks,
             number_of_exclamations,
             number_of_percentages,
             number_of_currency_symbols,
             number_of_paragraph_symbols,
             content_fraction,
             number_of_cappsed_words,
             number_of_first_person_pronouns]
        ).T

        return result

def filter_entities(entities_in, entity_filter = ["Person", "Location", "Organization", "Company"]):
    entities_out = []
    for val in entities_in:
        if val[0] in entity_filter:
            entities_out.append(val)
    return entities_out

def map_entities(entities):
    out = []
    for val in entities:
        out.append((val['type'], val['text'], val['relevance']))
    return out


def process_keywords(keywords):
    out = []
    for val in keywords:
        out.append((val['text'], val['relevance']))
    return out

def process_entities(entities):
    mapped_entites = map_entities(entities)# type : 'type', text: 'text', relevance: 'relevance' to (type, text relevance)
    filtered_entities = filter_entities( mapped_entites)
    return filtered_entities

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
    return title

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
        os.makedirs(output_path, exist_ok = True)
        with open(file_path, "wb") as jpgfile:
            jpgfile.write(jpg)

        #img = Image.open(file_path)
        #size = img.size
        #if size[0] < 100 or size[1] < 100:
        #    os.remove(file_path)
        #else:
        export_paths.append(file_path)
        njpg += 1

        i = iend

    return export_paths

def create_thumbnail(src_filename, output_path, pagenum = 0, resolution = 72,):
    src_pdf = PyPDF2.PdfFileReader(open(src_filename, "rb"))
    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(src_pdf.getPage(pagenum))
    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)
    img = Image(file = pdf_bytes, resolution = resolution)
    img.convert("png")
    print(output_path)
    export_path = output_path + "_thumb.png"
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

def vectorize_document(text, model):
    return np.array(getVectorsOf(model, text)).mean(axis=0)

def mongo_connect():
    client = MongoClient('localhost', 27017)
    db = client.stanford_data
    collection = db.document_collection
    documents = collection.documents
    return documents

def mongo_save(document, schema):
    doc = {"document_title_md": document['title_md'],
           "document_title_pdf": document['title_pdf'],
           "document_title_summ": summarize(document['text'], word_count=6, split=False),
           "document_text": document["text"],
         "document_summary": document['summary'],
         "thumbnail_path": document['thumbnail_path'],
         "extracted_image_paths": document['pdf_images_paths'],
         "document_url": document['url'],
         "document_parent_url": document['parent_url'],
         "document_type": document['document_type'],
         "filename": document["filename"],
         "keywords": document['keywords'], # word, score
         "entities": document['entities'], # entity, value, score, image
         "word2vec": document['vector_representation'].tolist(),
         "lingvector" : document['lingvector'].tolist(),
         "date": document['time']}
    schema.insert_one(doc)
    print("SUCCESS :)))")

def main(entity_limit = 50, keyword_limit = 20):
    schema = mongo_connect()
    with open('../notebooks/document_type_classifier.pkl', 'rb') as f:
        document_type_classifier = pickle.load(f)
    with open('../notebooks/url_features.pkl', 'rb') as f:
        url_features = pickle.load(f)

    model = api.load("glove-wiki-gigaword-300")  # download the model and return as object ready for use

    natural_language_understanding = NaturalLanguageUnderstandingV1(
        username='cccb5076-87bd-4992-b99e-29a0f258460b',
        password='Prop61GOuNtl',
        version='2018-03-16')

    start = 2000
    end = 3000

    for index, pdffile in enumerate(sorted(os.listdir('nextiterationhackathon2018/pdf'))[start:end]):
        try:
            print("Durchlauf " + str(start) + "/" + str(start+index) + "/" + str(end))
            if(pdffile.endswith(".json")):
                print("JSON found :(")
                continue

            pdfpath = os.path.join('nextiterationhackathon2018/pdf', pdffile)
            document = {}
            print("Titel start")
            # document title
            try:
                document['title_pdf'] = get_title_without_meta(pdfpath)
                document['title_md'] = get_title_from_meta(pdfpath)
            except pdfrw.errors.PdfParseError as e:
                print("PdfParseError")
                continue
            document['time'] = os.path.getmtime(pdfpath)
            # process raw text
            try:
                document['text'] = textract.process(pdfpath).decode("utf-8")
            except UnicodeDecodeError:
                print("UnicodeDecodeError")
                continue

            print("NLU start")


            #process entities and keywords
            response = natural_language_understanding.analyze(
            text=document['text'],
            features=Features(
                entities=EntitiesOptions(
                 sentiment=False,
                 limit=entity_limit),
                keywords=KeywordsOptions(
                  sentiment=False,
                  emotion=False,
                  limit=keyword_limit)))

            keywords = response['keywords']
            document['keywords'] = process_keywords(keywords)
            entities = response['entities']
            document['entities'] = process_entities(entities)

            # add metadata
            json_path = os.path.join('nextiterationhackathon2018/pdf', basename(pdffile) + '.json')
            with open(json_path, 'r+') as jsondata:
                metadata = json.load(jsondata)
                document['url'] = metadata['url']
                document['parent_url'] = metadata['parent_url']
                document['filename'] = document['url'].split('/')[-1]

            # document classification

            print("Classification start")

            generated_url_features = pd.DataFrame(columns=url_features)
            generated_url_features.loc[0] = np.zeros(len(url_features))
            url_feature = "tld-url" + "_" + '.'.join(tldextract.extract(document['url'])[:2])
            parent_url_feature = "tld-parent-url" + "_" + '.'.join(tldextract.extract(document['parent_url'])[:2])
            if url_feature in generated_url_features.columns:
                generated_url_features.loc[0][url_feature] = 1
            if parent_url_feature in generated_url_features.columns:
                generated_url_features.loc[0][parent_url_feature] = 1

            ling = LinguisticVectorizer()
            x_ling = ling.fit([document['text']]).transform([document['text']])
            document['lingvector'] = x_ling[0]
            ling_features = pd.DataFrame(x_ling, columns=ling.get_feature_names())

            w2v_features = pd.DataFrame([np.array(getVectorsOf(model, document["text"])).mean(axis=0)]).add_prefix("w2v_")
            features = pd.concat([generated_url_features, ling_features, w2v_features], axis=1)

            prediction = document_type_classifier.predict(features)

            document["document_type"] = prediction[0]

            print("Images start")

            outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", str(index))
            os.makedirs(outpath, exist_ok=True)

            # document images
            document['pdf_images_paths'] = create_pdf_images(pdfpath, os.path.join(outpath, "docimages"))
            document['thumbnail_path'] = create_thumbnail(pdfpath, os.path.join(outpath, "thumbnail"))

            print("Vector start")

            document['vector_representation'] = vectorize_document(document['text'], model)

            print("Summary start")
            document['summary'] = summarize(document['text'].replace('\n', ' '), word_count=100, split=False)

            print("Save start")
            mongo_save(document, schema)
        except Exception as e:
            print("EXCEPTION" + str(e))
            traceback.print_tb(e.__traceback__)
            continue

if __name__ == "__main__":
    main()
