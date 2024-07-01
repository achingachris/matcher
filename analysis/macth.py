# %% [markdown]
# # Introduction
#
# This notebook presents a generic LLM matchin algorithm which can be progressed as part of Swahilipot's matching of mentors and mentees, youth to opportunities etc. The algorithm has been developed to be as generic as possible, utilizing LLMs for identifying which fields
# to use in matching. It includes the following key components and being passed two dataframes:
#
# 1. An automatic analysis to determine fields to use for semantic search, locations and other types. This is controlled via an LLM prompt
# 2. A basic semantic search to get matches for each row in dataframe 1 in dataframe 2
# 3. For each match a distance is calculated using Google maps API to filter out matches which are too distant
# 4. For each remaining match, and LLM judge asseses the full record to filter out any which are not tru matches
#
# For step 1 an LLM is used, but once a `./data/matching_fields.json` is created it is used. This is also useful as it means you can manually adjust this file to include/exclude fields as-needed. This file is very important for the matching algorithm and can be a good place to tune to matching process.
#
# # Setup
#
# 1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) by selecting the installer that fits your OS version. Once it is installed you may have to restart your terminal (closing your terminal and opening again)
# 2. In this directory, open terminal
# 3. `conda env create -f environment.yml`
# 4. `conda activate matching-env`
# 5. Open this notebook in VS Code and use this environment
#
# 6. You will need mentor and mentee data ...
#
# - (Mentees)[https://docs.google.com/spreadsheets/d/1i8ItmzyVEi5H0tII1C-zEWKjJJjHmeTkfPA9MDLPSLA/edit#gid=0] and the later (Mentees 2.0)(https://docs.google.com/spreadsheets/d/1U82YEHpcusuCC39mgGd0C_yAhYKf15W6MEab8kab3eU/edit#gid=0)
# - (Mentors)[https://docs.google.com/spreadsheets/d/1dEi1bsScI-gyFcLlGe-UajqzEzSb4q79ArWKrSHM4jw/edit#gid=0]
#
# These were provided by Chris from Swahilipot and have had PII removed. Download these into this folder and confirm there is no PII.
#
# 7. You will need to copy `.env` to `.env.example` and set keys for APIs

# %% [markdown]
# # Libraries and functions

# %%
import os
import json
import sys
import pandas as pd
from time import sleep
import traceback

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.faiss import FAISS

from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

import uuid
import googlemaps

from dotenv import load_dotenv

load_dotenv("../.env")

MENTOR_DATA = "./data/mentors.xlsx"
MENTEE_DATA = "./data/mentee 2.0.xlsx"
DIST_MATRIX = "./data/dist_matrix.json"
MATCHING_FIELDS_JSON = "./data/matching_fields.json"
MATCHES_FILE = "./data/matches.xlsx"

# For distance calculation
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))


def setup_models():
    """
    Setup the models for the chat and embedding

    Returns:

    embedding_model: AzureOpenAIEmbeddings
    chat: AzureChatOpenAI

    """
    embedding_model = AzureOpenAIEmbeddings(
        deployment=os.getenv("OPENAI_TEXT_COMPLETION_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("BASE_URL"),
        chunk_size=16,
    )

    chat = AzureChatOpenAI(
        azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("BASE_URL"),
        temperature=0,
        max_tokens=1000,
    )

    return embedding_model, chat


# Stored in langchain_pg_collection and langchain_pg_embedding as this
def initialize_db(type, embedding_model):
    """
    Initialize the database for storing the embeddings

    Args:
    type: str: Type of DB to use. Options are "PGVECTOR" or "FAISS"
    embedding_model: AzureOpenAIEmbeddings: The embedding model to use

    Returns:
    db: PGVector or FAISS: The initialized database

    """

    db = {}

    COLLECTION_NAME = f"embedding"

    # Use Postgres DB if it has PGVector add-on
    if type == "PGVECTOR":
        CONNECTION_STRING = PGVector.connection_string_from_db_params(
            driver=os.environ.get("POSTGRES_DRIVER", "psycopg2"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ.get("POSTGRES_DB", "postgres"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        )
        db = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=embedding_model,
        )
    # Basic FAISS
    elif type == "FAISS":
        dimensions: int = len(embedding_model.embed_query("dummy"))
        db = FAISS(
            embedding_function=embedding_model,
            index=IndexFlatL2(dimensions),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=False,
        )
    else:
        print("Invalid DB type")
        sys.exit(1)

    return db


def add_document(content, metadata, db):
    """

    Add a document to the vecotr database along with its metadata

    Args:

    content: str: The content of the document
    metadata: dict: The metadata of the document
    db: PGVector or FAISS: The database to add the document to


    """

    if metadata is None:
        metadata = {}

    uuid_str = str(uuid.uuid4())
    metadata["custom_id"] = uuid_str

    new_doc = Document(page_content=content, metadata=metadata)
    id = db.add_documents([new_doc], ids=[uuid_str])


def read_data(filename):
    """

    Read data from an Excel file

    Args:

    filename: str: The name of the file to read

    Returns:

    data: pd.DataFrame: The data read from the file

    """
    data = pd.read_excel(filename)
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.lower()
    data["id"] = data.index
    data["id"] = data["id"].apply(lambda x: f"{filename}-{x}")
    print(f"{filename} > {data.shape}")
    return data


def call_llm(instructions, prompt, chat, retries=5, retry_count=0):
    """

    Call the LLM model

    Args:

    instructions: str: Instructions to provide to the model
    prompt: str: The prompt to provide to the model
    chat: AzureChatOpenAI: The chat model to use
    retries: int: Number of retries to attempt
    retry_count: int: Current retry count

    Returns:

    response: dict: The response from the model

    """
    messages = [
        SystemMessage(content=instructions),
        HumanMessage(content=prompt),
    ]
    try:
        response = chat(messages)
        response = response.content
        response = response.replace("```json", "").replace("```", "").replace("\n", "")
        try:
            response = json.loads(response)
        except:
            print("JSON didn't parse")
            print(response)
        return response
    except Exception as e:
        print("Error calling LLM")
        print(chat(messages))
        retry_count += 1
        if retry_count < retries:
            sleep(1)
            response = call_llm(
                instructions, prompt, chat, retries=retries, retry_count=retry_count
            )
            print(response)
            return response
        else:
            print("Error calling LLM. Max retries reached")
            return None


def matching_fields(data1, data2, chat, context_prompt, format_prompt="", force=False):
    '''

    Get the fields to use for matching. This generates a JSON record which drives how matching
    is done. It's controlled by prompts to determine fields types and output formats. Here are some examples ...

    fields_context_prompt = """
        You looking at two dataframes data1 and data2 to see which columns can be used for matching mentors to mentees
        Id fields like 'name', 'id' and 'email' CANNOT BE USED FOR MATCHING, exclude them in your response
        URL fields like 'url' cannot be used for matching
        Fields providing resume/CV locations cannot be used for matching
        Fields related to skills/interests and demographics can be used for matching
        Exclude fields related to strengths and weaknesses
        Exclude mentor fields related to goals
        IMPORTANT!!!! Exclude ward fields in location fields
        Always include area of interest fields
    """

    # Prompt used to decide output format when matching fields
    fields_format_prompt = """
    Please reply with a JSON record in this format:

        {
            "data1_fields": {
                "skills_fields": [<FIELD NAME>, ...],
                "location_fields": [<FIELD NAME>, ...],
                "demographics_fields": [<FIELD NAME>, ...],
                "preferred_mentorship_mode": [<FIELD NAME>, ...]
            },
            "data2_fields": {
                "skills_fields": [<FIELD NAME>, ...],
                "location_fields": [<FIELD NAME>, ...],
                "demographics_fields": [<FIELD NAME>, ...],
                "preferred_mentorship_mode": [<FIELD NAME>, ...]
            }
        }
    """

    Note: It creates a JSON file saved into ./data. If this file exists it will use it, otherwise it will regenerate.
    The force parameter will force a regeneration.

    Args:

    data1: pd.DataFrame: The first dataset
    data2: pd.DataFrame: The second dataset
    chat: AzureChatOpenAI: The chat model to use
    context_prompt: str: The context prompt to use, controls how fields are classified and matched
    format_prompt: str: The format prompt for the output
    force: bool: Whether to force re-creation of the matching fields

    Returns:

    resp: dict: The response from the model

    '''

    if os.path.exists(MATCHING_FIELDS_JSON) and not force:
        print("Found existing fields matching file! Using it ...")
        with open(MATCHING_FIELDS_JSON) as f:
            resp = json.load(f)
        print(json.dumps(resp, indent=4))
        return resp

    matches = []

    data1_cols = list(data1.columns)
    data2_cols = list(data2.columns)

    prompt = context_prompt
    prompt += f"Given these two records, what fields would be good for matching?"
    prompt += f"\n\nDATA1: {data1_cols}"
    prompt += f"\n\nDATA2: {data2_cols}"
    prompt += f"\n\nPlease reply with a JSON record in this format:"
    prompt += format_prompt

    resp = call_llm("", prompt, chat)
    print(json.dumps(resp, indent=4))

    with open(MATCHING_FIELDS_JSON, "w") as f:
        json.dump(resp, f, indent=4)

    return resp


def subset_data(data, fields):
    """

    Function to subset dataframe to a column list. Also drops empty columns.

    Args:

    data - Pandas dataframe
    fields - Columns to subset

    Returns:

    data - Pandas dataframe of subset data

    """

    fields = list(set(fields) & set(data.columns))
    data = data[fields]
    # Drop any columns which are all null
    data = data.dropna(axis=1, how="all")
    return data


def index_data(data_content, data_all, db):
    """

    Adds documents to the vector databse, with their embeddings.

    Args:

    data_content: Pandas dataframe, data to be indexed, usually a subset of columns
    data_all: Pandas data frame of the data but with all columns, used to store metadata for each embedding
    db: Vecotr DB, see initialize_db above
    """
    for index, row in data_all.iterrows():
        rec = data_content.iloc[index].to_dict()
        content = ""
        # Remove keys
        for r in rec:
            content += f"{rec[r]};"
        content = content.replace("\n", ";")
        metadata = row.to_dict()
        add_document(content, metadata, db)


def get_distance(addr1, addr2):
    key = f"{addr1}__{addr2}"
    with open(DIST_MATRIX) as f:
        dist_matrix = json.load(f)
    if key in dist_matrix:
        return dist_matrix[key]
    else:
        return -9999999


def calc_dist_matrix(addresses1, addresses2, skip_strings):
    """
    For two lists of unique addresses or location strings, calculates their distance to build
    a dictionary of address combinations and their distances, stored to file.

    Args:

    addresses1 - List of from addresses/location fields
    addresses2 - List of to addresses/location fields
    skip_strings - List of strings which force records to be skipped

    Returns:

    dist_matrix: Dictionary with key <addresses1>__<addresses2> and vlues of distance in km

    Dictionary also stored in ./data.
    """
    dist_matrix = {}
    for addr1 in list(set(addresses1)):
        for addr2 in list(set(addresses2)):
            skip = False
            if addr1 == "" or addr2 == "":
                skip = True
            for s in skip_strings:
                if s in addr1 or s in addr2:
                    skip = True
                    break
            if skip:
                continue
            key = f"{addr1}__{addr2}"
            print(key)
            if key not in dist_matrix:
                if addr1 == addr2 or "Mombasa Metropolitan" in addr2:
                    dist_matrix[key] = 0.0
                else:
                    dist = gmaps.distance_matrix(addr1, addr2, units="metric")["rows"][
                        0
                    ]["elements"][0]
                    if dist["status"] == "OK":
                        dist_matrix[key] = dist["distance"]["value"] / 1000.0
                        print(
                            f"        Distance between {addr1} and {addr2} is {dist_matrix[key]} km"
                        )
                    else:
                        dist_matrix[key] = None

    with open("dist_matrix.json", "w") as f:
        json.dump(dist_matrix, f, indent=4)

    return dist_matrix


def set_addresses(data, fields):
    """

    Sets an 'address' field on a dataframe, by concatenating vlaues in address 'fields'

    Args:

    data - Pandas dataframe
    fields - List of address-related fields

    Returns:

    data - Parndas dataframe with extra column 'address'

    """
    data_add = data[fields]
    addresses = data_add.apply(lambda x: ",".join(x.dropna().astype(str)), axis=1)
    addresses = addresses.str.lower()
    data["address"] = addresses
    return data


def get_ai_check_fields(match_fields):
    """

    Using the fields match JSON record, gets a list of all fields that might be useful for the AI
    match checker.

    Args:

    match_fields - dictionary of mathing fields, as output by function `match_fields`

    Returns:

    ai_check_fields - list of field names
    """

    ai_check_fields = []
    for k in match_fields:
        ai_check_fields += match_fields[k]
    print(ai_check_fields)
    return ai_check_fields


def check_pii(data, chat):
    """

    Check if data contains PII columns. Note, this is not perfect and should be used as a guide only.

    Args:

    data: pd.DataFrame: The data to check for PII
    prompt: str: The prompt to use to check for PII
    chat: AzureChatOpenAI: The chat model to use

    Returns:

    matching_fields_list: dict: The matching fields list

    """

    pii_prompt = """

    =================

    Analyze the text above 
    Please identify whether not the above text contains any instances where a person's name is given 
    Please identify all email addresses 
    I need to find any exact dates that correspond to the day a person was born. They typically follow the format MM/DD/YYYY or DD/YY or other combinations.  They Must have a day number. Can you find them in the text? 
    Find any locations that could be a person's home address in this text. They must have a stret number or apartment number.

    Your response should be a JSON record in this format:

    {
        "has_pii": <"yes" or "no">,
        "fields": "<FIELD>, ..."
        "values": "<VALUE>, ..."
    }

    """

    data = data.sample(5)
    data = data.to_dict(orient="records")
    data = json.dumps(data, indent=4)

    # Noting here we need a more powerful model to be safe
    chat = AzureChatOpenAI(
        azure_deployment="gpt-4-turbo",
        azure_endpoint=os.getenv("BASE_URL"),
        temperature=0,
        max_tokens=1000,
    )

    # Check data1 for PII
    print("\n\nChecking data for PII ...")
    prompt = data + pii_prompt
    print(prompt)

    resp = call_llm("", prompt, chat)

    if resp["has_pii"] == "yes":
        print("PII detected!")
        print(f"Fields: {resp['fields']}")
        print(f"Values: {resp['values']}")
        sys.exit()
    else:
        print("No PII detected")


def ai_check(data1, data2, chat, context_prompt=""):
    """

    Checks a match as found by semantic search to see it it's still a match when considering more fields

    Args:

    data1 - JSON record of record1, being a subset of input table row by ai_match_fields
    data2 - JSON record of record2, being a subset of input table row by ai_match_fields
    context_prompt String, prompt to guide the AI matcher further

    Returns:

    rep - dict, stating answer and reason
    """

    prompt = context_prompt
    prompt += f"Given these two records, what fields would be good for matching?"
    prompt += f"\n\nDATA1: {data1}"
    prompt += f"\n\nDATA2: {data2}"
    prompt += f"\n\nPlease reply with a JSON record in this format:"
    prompt += """
        {
            "answer": <"yes" or "no">,
            "reason": <"reason for answer">    
        }
    """
    resp = call_llm("", prompt, chat)
    print(json.dumps(resp, indent=4))
    return resp


def run_matching_batch(
    data1,
    data2,
    rec1_name,
    rec2_name,
    fields_context_prompt,
    fields_format_prompt,
    ai_check_prompt,
    max_travel_distance_km=15,
    data2_cap=10,
    force=False,
):
    """
    Main batch function for calculating matching between two dataframes using a staged approach (i) Simple vector similarity
    top k matches, then reviewed for distances between data, then finally assessed by AI match checker.

    Args:

    data1 - Pandas dataframe of first dataset (eg mentors)
    data2 - Pandas dataframe of second dataset to match against (eg mentees)
    rec1_name - String name of first dataset, eg 'mentors'
    rec2_name - String name of second dataset, eg 'menteess'
    fields_context_prompt: str: The context prompt to use, controls how fields are classified and matched, see 'matching_fields' function
    fields_format_prompt: str: The format prompt for the output, see 'matching_fields' function
    ai_check_prompt: str, prompt to guide the AI matcher further, see ai_check function
    max_traveL_distance_km: Maximum travel distance allowed for a match. If exceeded, match field <rec1_name>_distance_match is set to 'no'
    data2_cap: Number of data2 records to try and match with all of data1
    force: Bool, used to force refresh of matching_fields JSON record

    Returns:

    matches: Pandas dataframe of each match with all data1 fields plus extra columns for the matched data2 fields

    Note, the matches willl have a row per match, so mentees values are repeated for each mentor. The function also saves
    matches to an excel file in ./data

    """

    print("\n=====> Setting up models ...")
    embedding_model, chat = setup_models()
    db = initialize_db("FAISS", embedding_model)

    print("\n=====> Identifying match fields ...")
    matching_fields_list = matching_fields(
        data1, data2, chat, fields_context_prompt, fields_format_prompt, force=force
    )

    # Please do not comment this out. :)
    print("\n=====> Checking for PII  ...")
    check_pii(data1, chat)
    check_pii(data2, chat)

    print("\n=====> Calculating travel distance matrix ...")
    data1 = set_addresses(
        data1, matching_fields_list["data1_fields"]["location_fields"]
    )
    data2 = set_addresses(
        data2, matching_fields_list["data2_fields"]["location_fields"]
    )
    dist_matrix = calc_dist_matrix(
        data1["address"].unique(),
        data2["address"].unique(),
        skip_strings=["outside mombasa"],
    )

    print("\n=====> Calculating index for semantic match ...")
    data1_search = subset_data(
        data1, matching_fields_list["data1_fields"]["skills_fields"]
    )
    data2_search = subset_data(
        data2, matching_fields_list["data2_fields"]["skills_fields"]
    )
    print(list(data1_search.columns))
    data1_index = index_data(data1_search, data1, db)

    print("\n=====> Extracting AI check fields for datasets ...")
    ai_check_fields1 = get_ai_check_fields(matching_fields_list["data1_fields"])
    ai_check_fields2 = get_ai_check_fields(matching_fields_list["data2_fields"])

    print("\n=====> Starting batch matching of mentees ...")
    matches = []
    for index, row in data2[0:data2_cap].iterrows():

        try:

            # Simple semantic search
            data2_search_fields = list(data2_search.iloc[index])
            data2_search_string = " ".join(data2_search_fields)
            print(f"\n\n{rec2_name}: {data2_search_string}")
            print(f"{rec2_name} ID: {row['id']}")
            print(f"{rec2_name} Address: {row['address']}")
            docs_and_scores = db.similarity_search_with_score(
                data2_search_string, top_k=3
            )
            print(f"RESULTS: {len(docs_and_scores)}")

            data2_address = row["address"]
            data2_ai_record = data2.iloc[index][ai_check_fields2].to_dict()
            data2_ai_record = json.dumps(data2_ai_record, indent=4)
            print(f"{rec2_name} AI Fields:\n")
            print(data2_ai_record)

            # Filter simple results by secondary filters (distance and ai check)
            data2["matches"] = ""
            for d in docs_and_scores:

                data1_skills = d[0].page_content
                data1_full = d[0].metadata

                # Flag if distances are too far
                data1_address = data1_full["address"]
                travel_distance_km = get_distance(data1_address, data2_address)
                distance_match = "yes"
                if travel_distance_km > max_travel_distance_km:
                    print("Match found but distance too far")
                    distance_match = "no"

                print(f"Score: {d[1]}")
                print(f"{rec1_name} ID: {data1_full['id']}")
                print(f"{rec1_name} match terms: {data1_skills}")
                print(f"{rec1_name} location: {data1_full['address']}")
                print(f"Distance to {rec1_name}: {travel_distance_km} km")

                data1_ai_record = pd.DataFrame([data1_full])
                data1_ai_record = data1_ai_record[ai_check_fields1]
                data1_ai_record = json.loads(data1_ai_record.to_json(orient="records"))[
                    0
                ]
                data1_ai_record = json.dumps(data1_ai_record, indent=4)
                print("AI Check fields:")
                print(f"   {rec2_name} : {data2_ai_record}")
                print(f"   {rec1_name}: {data1_ai_record}")
                print("******* AI CHECK:")
                ai_check_result = ai_check(
                    data1_ai_record, data2_ai_record, chat, ai_check_prompt
                )

                # Build match record
                r = {}
                r["id"] = row["id"]
                r["address"] = data2_address
                r[f"{rec1_name}_id"] = data1_full["id"]
                for f in json.loads(data1_ai_record):
                    r[f"{rec1_name}_{f}"] = data1_full[f]
                r[f"{rec1_name}_address"] = data1_address
                r[f"{rec1_name}_score"] = d[1]
                r[f"{rec1_name}_match_terms"] = data1_skills
                r[f"{rec1_name}_distance"] = travel_distance_km
                r[f"{rec1_name}_json_comparison"] = (
                    f"MENTEE:\n{data2_ai_record}\n\nMENTOR:\n{data1_ai_record}"
                )
                r[f"{rec1_name}_distance_match"] = distance_match
                r[f"{rec1_name}_ai_check_result"] = ai_check_result["answer"]
                r[f"{rec1_name}_ai_check_reason"] = ai_check_result["reason"]
                matches.append(r)

        except Exception as e:
            print(f"Error processing record {index}")
            print(traceback.format_exc())
            print(e)

    if len(matches) > 0:
        matches = pd.DataFrame(matches)
        data2 = data2.merge(matches, on="id", how="left")
        print(f"Saving matches to {MATCHES_FILE}")
        data2.to_excel(MATCHES_FILE, index=False)

    return data2


# %% [markdown]
# ## Matching batch for mentors and mentees

# %%
print("Reading in two dataframes to match ...")
mentors = read_data(MENTOR_DATA)
# print(mentors.columns)
mentees = read_data(MENTEE_DATA)
# print(mentees.columns)

# Prompt used when automatically identifying matching fields
fields_context_prompt = """
    You looking at two dataframes data1 and data2 to see which columns can be used for matching mentors to mentees
    Id fields like 'name', 'id', 'address' and 'email' CANNOT BE USED FOR MATCHING, exclude them in your response
    URL fields like 'url' cannot be used for matching
    Fields providing resume/CV locations cannot be used for matching
    Fields related to skills/interests and demographics can be used for matching
    Exclude fields related to strengths and weaknesses
    Exclude mentor fields related to goals
    IMPORTANT!!!! Exclude ward fields in location fields
    Always include area of interest fields
"""

# Prompt used to decide output format when matching fields
fields_format_prompt = """
Please reply with a JSON record in this format:

        {
            "data1_fields": {
                "skills_fields": [<FIELD NAME>, ...],
                "location_fields": [<FIELD NAME>, ...],
                "demographics_fields": [<FIELD NAME>, ...],
                "preferred_mentorship_mode": [<FIELD NAME>, ...]
            },
            "data2_fields": {
                "skills_fields": [<FIELD NAME>, ...],
                "location_fields": [<FIELD NAME>, ...],
                "demographics_fields": [<FIELD NAME>, ...],
                "preferred_mentorship_mode": [<FIELD NAME>, ...]
            }
        }
"""

ai_check_prompt = """
    You are an AI checking potential matches between mentors and mentees
    You ignore locations and focus on other fields
    If mentor or mentee wants similar demographics, such as gender, age etc, be very careful comparing those fields
    Given the above, is this mentor matched to this mentee?
"""

max_travel_distance_km = 15
mentee_cap = 10

matches = run_matching_batch(
    mentors,
    mentees,
    "mentor",
    "mentee",
    fields_context_prompt,
    fields_format_prompt,
    ai_check_prompt,
    max_travel_distance_km,
    mentee_cap,
)


# %%
