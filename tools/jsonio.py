import json

def _save(record,filename):
    json_dict = record

    json_str = json.dumps(json_dict, indent=4)
    with open(filename, 'w') as json_file:
        json_file.write(json_str)

def _load(filename):
    f = open(filename)
    data = json.load(f)
    f.close()
    return data