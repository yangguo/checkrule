import os

import pandas as pd
import pymongo
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.environ.get("MONGO_DB_URL")

# Connect to the MongoDB
def get_mongo_remote():
    client = pymongo.MongoClient(MONGO_DB_URL)
    # db = client.test
    # Send a ping to confirm a successful connection
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client


def get_collection(dbname, collectionname):
    # client = get_mongo_client()
    client = get_mongo_remote()
    db = client[dbname]
    collection = db[collectionname]
    return collection


# insert dataframes into MongoDB
def insert_data(df, collection):
    records = df.to_dict("records")
    collection.insert_many(records)


# get dataframes from MongoDB
def get_data(collection):
    # get data from MongoDB
    df = pd.DataFrame(list(collection.find()))
    return df


# delete dataframes from MongoDB
def delete_data(collection):
    collection.delete_many({})
