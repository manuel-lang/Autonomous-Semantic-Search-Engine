from neo4j.v1 import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from pymongo import MongoClient
import re
from contextlib import suppress

client = MongoClient("localhost", 27017)
db = client.stanford_data
# collection = db.document_collection
collection = db
documents = collection.documents

keyword_set = set()
organization_set = set()
location_set = set()
company_set = set()
person_set = set()

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123"))
with driver.session() as session:
    for document in documents.find():
        print(document)  # iterate the cursor
        with session.begin_transaction() as tx:
            match_queries = []
            queries = []
            entities = []
            keywords = []
            counter = 0
            for entity in document["entities"]:
                entity_r = entity[1]  # re.sub('[^A-Za-z0-9]+', '', entity[1])
                if entity[0] == "Organization":
                    if entity_r not in organization_set:
                        queries.append("(e" + str(counter) + ":Organization { name: \"" + entity_r + "\"})")
                        organization_set.add(entity_r)
                    else:
                        match_queries.append(
                            "MATCH (e" + str(counter) + ":Organization) WHERE e" + str(counter) + ".name = \"" + entity_r + "\"\n")
                    entities.append("e" + str(counter))
                    counter += 1
                elif entity[0] == "Location":
                    if entity_r not in location_set:
                        queries.append("(e" + str(counter) + ":Location { name: \"" + entity_r + "\"})")
                        location_set.add(entity_r)
                    else:
                        match_queries.append(
                            "MATCH (e" + str(counter) + ":Location) WHERE e" + str(counter) + ".name = \"" + entity_r + "\"\n")
                    entities.append("e" + str(counter))
                    counter += 1
                elif entity[0] == "Company":
                    if entity_r not in company_set:
                        queries.append("(e" + str(counter) + ":Company { name: \"" + entity_r + "\"})")
                        company_set.add(entity_r)
                    else:
                        match_queries.append(
                            "MATCH (e" + str(counter) + ":Company) WHERE e" + str(counter) + ".name = \"" + entity_r + "\"\n")
                    entities.append("e" + str(counter))
                    counter += 1
                elif entity[0] == "Person":
                    if entity_r not in person_set:
                        queries.append("(e" + str(counter) + ":Person { name: \"" + entity_r + "\"})")
                        person_set.add(entity_r)
                    else:
                        match_queries.append(
                            "MATCH (e" + str(counter) + ":Person) WHERE e" + str(counter) + ".name = \"" + entity_r + "\"\n")
                    entities.append("e" + str(counter))
                    counter += 1
            for keyword in document["keywords"]:
                keyword_r = keyword[0]  # re.sub('[^A-Za-z0-9]+', '', keyword[0])
                if not keyword_r in keyword_set:
                    queries.append("(e" + str(counter) + ":Topic { name: \"" + re.sub('[^A-Za-z0-9]+', '', keyword_r) + "\"})")
                    keyword_set.add(keyword_r)
                else:
                    match_queries.append("MATCH (e" + str(counter) + ":Topic) WHERE e" + str(counter) + ".name = \"" + entity_r + "\"\n")
                keywords.append("e" + str(counter))
                counter += 1

            queries.append("(er:Document {name: \"" + re.sub('[^A-Za-z0-9]+', '', document["document_title_pdf"]) + "\"})")

            # for entity in entities:
            for entity in entities:
                queries.append("(" + entity + ")-[:r]->(er)")
            for k in keywords:
                queries.append("(" + k + ")-[:r]->(er)")

            try:
                tx.run("".join(match_queries) + " CREATE " + ",".join(queries))
            except:
                pass



print("finish")




#
# documents = []
#
# with driver.session() as session:
#     for document in documents:
#         with session.begin_transaction() as tx:
#             for record in tx.run("MATCH (a:Person)-[:KNOWS]->(f) "
#                                  "WHERE a.name = {name} "
#                                  "RETURN f.name", name=name):
#                 print(record["f.name"])
#
# driver.close()
