# %%
import json
import pickle
import time

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

settings = {}

with open("settings.json", "r") as f:
    settings = json.load(f)

codes_list = []

for i in range(1, 100):
    tmp_codes_list = api.kernels_list(competition=settings["COMPETITION_NAME"], page=i, page_size=100)

    if not bool(tmp_codes_list):
        break

    codes_list += tmp_codes_list

    time.sleep(1)

with open("codes_list.pkl", "wb") as f:
    pickle.dump(codes_list, f)

# %%
