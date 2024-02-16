from fastapi import FastAPI, File, UploadFile
import uvicorn
import pathlib
import Main as Mf

UPLOAD_FOLDER = "./testfile"
ALLOWED_EXTENSIONS = {"wav"}

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World 2"}


import os
import zipfile
from dev_config import test_file_path


@app.post("/speechrecognition")
async def create_upload_file(file: UploadFile = File(...)):
    # filenames = os.listdir("testfile/")
    id = []
    # SurahVerse = []
    # word=[]
    # distance = []
    # status = []
    # chunks_timestampes=[]
    sample_rate = 16000
    # for filename in filenames:
    # file_location = f"test/test.wav"
    print("file", file)
    print("test_file_path", test_file_path)

    with open(test_file_path, "wb+") as file_object:
        file_object.write(file.file.read())
    Surah_Result, wordsresult, distance1, status1, chunks_timestamps = Mf.Main(
        test_file_path
    )
    
    # SurahVerse.append(Surah_Result)
    # word.append(wordsresult)
    # distance.append(distance1)
    # status.append(status1)
    # chunks_timestampes.append(chunks_timestamps)
    # wordsresults=[]
    # score=[]
    start = []
    end = []
    # for i in range(len(wordsresult)):
    #    a=str(wordsresult[i])[3:-2]
    #    x=a.split(", ")
    #    print(x)
    #    wordsresults.append(x[0])
    #    score.append(x[1])

    for i in range(len(chunks_timestamps)):
        a = chunks_timestamps[i]
        start.append(a[0])
        end.append(a[1])
    finalresult = []
    lenghths = [
        len(wordsresult),
        len(start),
        len(start),
        len(start),
        len(start),
        len(distance1),
    ]
    wordssize = min(lenghths)
    for i in range(wordssize):
        finalresult.append(
            {
                "word": wordsresult[i],
                "starttime": start[i],
                "endtime": end[i],
                "score": 1,
                "distance": distance1[i],
            }
        )
    results_final = [
        {
            "surahverse": Surah_Result,
            "status": status1,
            "combinewordsresult": finalresult,
            # 'words': wordsresults,vfdvrve
            # 'score': score,
            # 'distance': distance1,
            # 'starttime':start,
            # 'endtime':end
        }
    ]
    print(results_final)

    return results_final


if __name__ == "__main__":
    # Use the below line if you want to specify the port and/or host
    uvicorn.run(app, host="0.0.0.0", port=8000)
