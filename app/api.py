import requests
import json


def fetch_speeches():

    # Open JSON file
    f = open('data/speeches.json')

    data = list(json.load(f))

    # Iterating through the json list
    speeches = []
    i = 0
    for i in range(len(data)):
        if i > 3:
            break

        speeches.append(data[i])

        i += 1

    # Close file
    f.close()

    print(json.dumps(speeches))

    # Return list of speeches
    return speeches