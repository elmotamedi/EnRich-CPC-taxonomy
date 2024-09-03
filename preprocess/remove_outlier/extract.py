import pandas as pd
import os
import signal
import json
import zstandard as zstd
from multiprocessing import Pool

N_THREADS = 24

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def values_for_field(item, key):
    values = {}
    for ii in range(len( item['description_localized'])):
        try:
            v =  item[key][ii]['text'].strip()
            lang = item[key][ii]["language"].strip()

            if not v:
                continue
            if not lang:
                continue
            values[lang] = v
        except:
            continue
    return values

def process_file(paths):
    inpath, outpath = paths
    data = pd.read_parquet(inpath, engine='pyarrow')
    entries = []
    for _, item in data.iterrows():
        cpc_codes = []
    
        
        if len(item['cpc']) == 0 :
            continue
        if len(item['description_localized']) == 0 and len(item["abstract_localized"]) == 0:
            continue

        # print(item.keys())

        cpc_codes = []
        for cpc in item['cpc']:
            cpc_codes.append(cpc['code'])
        pn = item["publication_number"]
        an = item["application_number"]


        abstracts = values_for_field(item, "abstract_localized")
        descriptions = values_for_field(item, 'description_localized')
        titles = values_for_field(item, "title_localized")

        all_langs = list(abstracts.keys()) + list(descriptions.keys()) + list(titles.keys())
        all_langs = list(set(all_langs))

        for lang in all_langs:
            abstract = abstracts[lang] if lang in abstracts else None
            description = descriptions[lang] if lang in descriptions else None
            title = titles[lang] if lang in titles else None
            if abstract is None and description is None:
                continue

            entry = {
                "publication_number": pn,
                "application_number": an,
                "cpc_code": cpc_codes,
                "lang": lang,
                "title": title,
                "abstract": abstract,
                "text": description,
            }
            entries.append(entry)
    if len(entries) == 0:
        return
    
    cctx = zstd.ZstdCompressor()
    with open(outpath, 'wb') as file:
        with cctx.stream_writer(file) as compressor:
            for item in entries:
                json_line = json.dumps(item) + '\n'
                compressor.write(json_line.encode('utf-8'))
    
    return


def process_files(outputpath, inputpath):
    files_inp = [f for f in os.listdir(inputpath) if f.endswith(".parquet")]
    files_out = [os.path.splitext(f)[0] + ".jsonl.z" for f in files_inp]

    files_inp = [os.path.join(inputpath, f) for f in files_inp]
    files_out = [os.path.join(outputpath, f) for f in files_out]

    files = list(zip(files_inp, files_out))
    pool = Pool(N_THREADS, initializer=init_worker)
    try:
        pool.map(process_file, files)
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.close()
        pool.join()
    


if __name__ == "__main__":
    outputpath = "outputs/Loutputs/"
    os.makedirs(outputpath, exist_ok=True)
    # inputpath="data/"
    inputpath ="data/"
    process_files(outputpath, inputpath)
