import csv
import collections

DATASET_PATH = "./datasets/"
WIKI_FILE_NAME = "wikiElec.ElecBs3.txt"
TSV_FILE_NAME = "wiki.tsv"
TSV_FIELDS = ["FromNodeId", "ToNodeId", "Sign"]

def tokenize_wiki():
    wiki_file = open(DATASET_PATH + WIKI_FILE_NAME, encoding="latin-1")
    lines = wiki_file.readlines()
    
    res = []
    current = []
    for line in lines:
        if line.startswith("#"):
            continue
        if line != "\n":
            current.append(line.strip().split("\t"))
        else:
            res.append(current)
            current = []
    if current:
        res.append(current)
    return res

def normalize_wiki():
    raw_votes = tokenize_wiki()
    res = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for vote_entries in raw_votes:
        voted_user = vote_entries[2][1]
        for vote_result in [l for l in vote_entries if l[0] == "V"]:
            voter_id = vote_result[2]
            choice = vote_result[1]
            res[voter_id][voted_user] += int(choice)

    return res

def write_wiki_file(normalized_wiki):
    with open(DATASET_PATH + TSV_FILE_NAME, "w") as csvfile:
        fieldnames = TSV_FIELDS
        writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=fieldnames)

        writer.writeheader()
        
        for voter_id, voter_votes in normalized_wiki.items():
            for voted_user, aggregated_choice in voter_votes.items():
                if aggregated_choice == 0:
                    continue
                writer.writerow({TSV_FIELDS[0]: voter_id, TSV_FIELDS[1]: voted_user, TSV_FIELDS[2]: 1 if aggregated_choice > 0 else -1})

def create_wiki_tsv_file():
    normalized = normalize_wiki()
    write_wiki_file(normalized)

    