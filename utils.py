import json

def get_params(filepath="extraction_params.json"):
    with open(filepath, "r") as f:
        return json.load(f)
