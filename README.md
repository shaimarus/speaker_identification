# Speaker Identification using speechbrain/spkrec-ecapa-voxceleb (for embeddings) and hnswlib (for fast calc cosine distance)
1.Change the name of index_path=/home + path to the index folder.  <br/>

2.Navigate to the docker images folder and then run docker-compose build up -d (if you want to work offline, you need to build the images first and then load them).  <br/>

3.It is necessary to gather the voice database (localhost:8006/get_vector_from_audio_and_insert_to_db/home/+file path+/data/id04536_2j8I_WX5mhY_00028.wav).  <br/>

4.Launch a search in the database (localhost:8006/speaker_indetification/home/+file path+/data/id04536_2j8I_WX5mhY_00028.wav).  <br/>

5.When searching, you may encounter the error "cannot return the results in a contiguous 2D array. Probably ef or M is too small" due to the small size of the voice database.  <br/>

Important! Do not touch the file_name_vectors_all table! If you delete rows or truncate it, also delete hnswlib.index, otherwise the search will not work correctly (indexes will be lost).  <br/>



![Image alt](https://github.com/shaimarus/speaker_identification/blob/main/finder.png)
