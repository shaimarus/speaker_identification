from fastapi import FastAPI
import uvicorn
from speechbrain.pretrained import EncoderClassifier
import torchaudio
import psycopg2
import pandas as pd
import numpy as np
import hnswlib
import os

from fastapi.middleware.cors import CORSMiddleware

index_path = os.environ["index_path"]

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



conn = psycopg2.connect(
    database="postgres",
    user='postgres',
    password='postgres',
    host='db',
    port='5432'
)
conn.autocommit = True
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS file_name_vectors_all(
file_name  varchar(1000) NOT NULL,
x1 numeric not null,
x2 numeric not null,
x3 numeric not null,
x4 numeric not null,
x5 numeric not null,
x6 numeric not null,
x7 numeric not null,
x8 numeric not null,
x9 numeric not null,
x10 numeric not null,
x11 numeric not null,
x12 numeric not null,
x13 numeric not null,
x14 numeric not null,
x15 numeric not null,
x16 numeric not null,
x17 numeric not null,
x18 numeric not null,
x19 numeric not null,
x20 numeric not null,
x21 numeric not null,
x22 numeric not null,
x23 numeric not null,
x24 numeric not null,
x25 numeric not null,
x26 numeric not null,
x27 numeric not null,
x28 numeric not null,
x29 numeric not null,
x30 numeric not null,
x31 numeric not null,
x32 numeric not null,
x33 numeric not null,
x34 numeric not null,
x35 numeric not null,
x36 numeric not null,
x37 numeric not null,
x38 numeric not null,
x39 numeric not null,
x40 numeric not null,
x41 numeric not null,
x42 numeric not null,
x43 numeric not null,
x44 numeric not null,
x45 numeric not null,
x46 numeric not null,
x47 numeric not null,
x48 numeric not null,
x49 numeric not null,
x50 numeric not null,
x51 numeric not null,
x52 numeric not null,
x53 numeric not null,
x54 numeric not null,
x55 numeric not null,
x56 numeric not null,
x57 numeric not null,
x58 numeric not null,
x59 numeric not null,
x60 numeric not null,
x61 numeric not null,
x62 numeric not null,
x63 numeric not null,
x64 numeric not null,
x65 numeric not null,
x66 numeric not null,
x67 numeric not null,
x68 numeric not null,
x69 numeric not null,
x70 numeric not null,
x71 numeric not null,
x72 numeric not null,
x73 numeric not null,
x74 numeric not null,
x75 numeric not null,
x76 numeric not null,
x77 numeric not null,
x78 numeric not null,
x79 numeric not null,
x80 numeric not null,
x81 numeric not null,
x82 numeric not null,
x83 numeric not null,
x84 numeric not null,
x85 numeric not null,
x86 numeric not null,
x87 numeric not null,
x88 numeric not null,
x89 numeric not null,
x90 numeric not null,
x91 numeric not null,
x92 numeric not null,
x93 numeric not null,
x94 numeric not null,
x95 numeric not null,
x96 numeric not null,
x97 numeric not null,
x98 numeric not null,
x99 numeric not null,
x100 numeric not null,
x101 numeric not null,
x102 numeric not null,
x103 numeric not null,
x104 numeric not null,
x105 numeric not null,
x106 numeric not null,
x107 numeric not null,
x108 numeric not null,
x109 numeric not null,
x110 numeric not null,
x111 numeric not null,
x112 numeric not null,
x113 numeric not null,
x114 numeric not null,
x115 numeric not null,
x116 numeric not null,
x117 numeric not null,
x118 numeric not null,
x119 numeric not null,
x120 numeric not null,
x121 numeric not null,
x122 numeric not null,
x123 numeric not null,
x124 numeric not null,
x125 numeric not null,
x126 numeric not null,
x127 numeric not null,
x128 numeric not null,
x129 numeric not null,
x130 numeric not null,
x131 numeric not null,
x132 numeric not null,
x133 numeric not null,
x134 numeric not null,
x135 numeric not null,
x136 numeric not null,
x137 numeric not null,
x138 numeric not null,
x139 numeric not null,
x140 numeric not null,
x141 numeric not null,
x142 numeric not null,
x143 numeric not null,
x144 numeric not null,
x145 numeric not null,
x146 numeric not null,
x147 numeric not null,
x148 numeric not null,
x149 numeric not null,
x150 numeric not null,
x151 numeric not null,
x152 numeric not null,
x153 numeric not null,
x154 numeric not null,
x155 numeric not null,
x156 numeric not null,
x157 numeric not null,
x158 numeric not null,
x159 numeric not null,
x160 numeric not null,
x161 numeric not null,
x162 numeric not null,
x163 numeric not null,
x164 numeric not null,
x165 numeric not null,
x166 numeric not null,
x167 numeric not null,
x168 numeric not null,
x169 numeric not null,
x170 numeric not null,
x171 numeric not null,
x172 numeric not null,
x173 numeric not null,
x174 numeric not null,
x175 numeric not null,
x176 numeric not null,
x177 numeric not null,
x178 numeric not null,
x179 numeric not null,
x180 numeric not null,
x181 numeric not null,
x182 numeric not null,
x183 numeric not null,
x184 numeric not null,
x185 numeric not null,
x186 numeric not null,
x187 numeric not null,
x188 numeric not null,
x189 numeric not null,
x190 numeric not null,
x191 numeric not null,
x192 numeric not null
)"""
)

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")




@app.get("/get_vector_from_audio_and_insert_to_db/{file_name:path}")
async def get_info(file_name):

    #####add this file to to db
    signal, _ =torchaudio.load(file_name)
    embeddings = classifier.encode_batch(signal)
    vector=embeddings[0][0].tolist()
    vector1=[file_name]+vector
    cursor.execute("INSERT into file_name_vectors_all(file_name, x1,x2,	x3,	x4,	x5,	x6,	x7,	x8,	x9,	x10,	x11,	x12,	x13,	x14,	x15,	x16,	x17,	x18,	x19,	x20,	x21,	x22,	x23,	x24,	x25,	x26,	x27,	x28,	x29,	x30,	x31,	x32,	x33,	x34,	x35,	x36,	x37,	x38,	x39,	x40,	x41,	x42,	x43,	x44,	x45,	x46,	x47,	x48,	x49,	x50,	x51,	x52,	x53,	x54,	x55,	x56,	x57,	x58,	x59,	x60,	x61,	x62,	x63,	x64,	x65,	x66,	x67,	x68,	x69,	x70,	x71,	x72,	x73,	x74,	x75,	x76,	x77,	x78,	x79,	x80,	x81,	x82,	x83,	x84,	x85,	x86,	x87,	x88,	x89,	x90,	x91,	x92,	x93,	x94,	x95,	x96,	x97,	x98,	x99,	x100,	x101,	x102,	x103,	x104,	x105,	x106,	x107,	x108,	x109,	x110,	x111,	x112,	x113,	x114,	x115,	x116,	x117,	x118,	x119,	x120,	x121,	x122,	x123,	x124,	x125,	x126,	x127,	x128,	x129,	x130,	x131,	x132,	x133,	x134,	x135,	x136,	x137,	x138,	x139,	x140,	x141,	x142,	x143,	x144,	x145,	x146,	x147,	x148,	x149,	x150,	x151,	x152,	x153,	x154,	x155,	x156,	x157,	x158,	x159,	x160,	x161,	x162,	x163,	x164,	x165,	x166,	x167,	x168,	x169,	x170,	x171,	x172,	x173,	x174,	x175,	x176,	x177,	x178,	x179,	x180,	x181,	x182,	x183,	x184,	x185,	x186,	x187,	x188,	x189,	x190,	x191,x192) VALUES (%s, %s,%s,%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s,	%s)", vector1)

    #### add index for this file
    
    index = hnswlib.Index(space = 'cosine', dim = 192)
    

    df = pd.read_sql_query('select file_name from "file_name_vectors_all"',con=conn)

    if not os.path.exists(index_path):

        index.init_index(max_elements = 1, ef_construction = 400, M = 64)
        index.set_ef(50)
        index.add_items(np.array(vector), list(range(1)))
        index.save_index(index_path)

        
    else:
        index.load_index(index_path, max_elements = len(df)+1)
        index.add_items(np.array(vector))
        index.save_index(index_path)
    
    
    
from fastapi.responses import ORJSONResponse
@app.get("/speaker_indetification/{file_name:path}", response_class=ORJSONResponse)
async def get_info(file_name):

    signal, _ =torchaudio.load(file_name)
    embeddings = classifier.encode_batch(signal)
    vector=embeddings[0][0].tolist()

    index = hnswlib.Index(space = 'cosine', dim = 192)
    index.load_index(index_path)
    top_k_hits=10
    corpus_ids, distances = index.knn_query(vector, k=top_k_hits)

    hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)    
    
    df = pd.read_sql_query('select file_name from "file_name_vectors_all"',con=conn)

    return ORJSONResponse({'file_name':[df['file_name'][hit['corpus_id']] for hit in hits[0:top_k_hits] if hit['score']>0.1],
                           'score':[hit['score'] for hit in hits[0:top_k_hits] if hit['score']>0.1]})


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True, workers=2)







