import requests
import json

def fetch_products(app):
    """Grab product listings from BestBuy."""
    # endpoint = "https://api.bestbuy.com/v1/products(customerReviewAverage>=4&customerReviewCount>100&longDescription=*)"
    # params = {
    #     "show": "customerReviewAverage,customerReviewCount,name,sku,image,description,manufacturer,longDescription,salePrice,sku",
    #     "apiKey": app.config["BEST_BUY_API_KEY"],
    #     "format": "json",
    #     "pageSize": 6,
    #     "totalPages": 1,
    #     "sort": "customerReviewAverage.dsc",
    # }
    # headers = {"Accept": "application/json", "Content-Type": "application/json"}
    # req = requests.get(endpoint, params=params, headers=headers)
    # products = req.json()["products"]

    products1 = [
        {
        "name": "Fake Name",
        "image": "http://google.com/image.png",
        "salePrice": "1000"
        }
    ]

    return products1

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