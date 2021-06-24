import json
import pickle
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from google.cloud import storage as gcs

# from google.oauth2 import service_account
# key_path = 'GCP_kaggle_slack_notify_sa_key.json'
# credential = service_account.Credentials.from_service_account_file(key_path)


def kaggle_notify(event, context):
    project_id = "project_id"
    bucket_name = "backet_name"

    client = gcs.Client(project_id)
    # client = gcs.Client(project_id, credentials=credential)
    bucket = client.get_bucket(bucket_name)

    gcs_path = "settings.json"
    blob = bucket.blob(gcs_path)
    settings = json.loads(blob.download_as_string().decode('utf-8'))

    api = KaggleApi()
    api.authenticate()

    codes_list = []

    for i in range(1, 100):
        tmp_codes_list = api.kernels_list(competition=settings["COMPETITION_NAME"], page=i, page_size=100)

        if not bool(tmp_codes_list):
            break

        codes_list += tmp_codes_list

        time.sleep(1)

    gcs_path = "codes_list.pkl"
    blob = bucket.blob(gcs_path)
    old_codes_list = pickle.loads(blob.download_as_string())

    old_codes_list_title = [x.title for x in old_codes_list]

    newly_posted_codes_list = []

    for code in codes_list:
        if code.title not in old_codes_list_title:
            newly_posted_codes_list.append(code)

    slacl_api = "https://slack.com/api/chat.postMessage"

    for code in newly_posted_codes_list:
        data = {
            "token": settings["SLACK_TOKEN"],
            "channel": settings["SLACK_CHANNEL_NAME"],
            "text": f"<@yshimizu765>\n新規のコード投稿がありました\ntitle : {code.title}\ntotalVotes : {code.totalVotes}\nlink : https://www.kaggle.com/{code.ref}"
        }

        requests.post(slacl_api, data=data)

        time.sleep(0.1)
    
    gcs_path = "codes_list.pkl"
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(pickle.dumps(codes_list))


# if __name__ == "__main__":
#     kaggle_notify(None, None)
