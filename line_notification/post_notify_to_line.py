import requests, sys
PATH = './'

def get_result_diff(result, pastResult):
  with open(result) as f:
    result_list = f.readlines()[2:-1]
    result_list = set(x.split("  ")[0] for x in result_list)
  
  with open(pastResult) as f:
    pastResult_list = f.readlines()[2:-1]
    pastResult_list = set(x.split(" ")[0] for x in pastResult_list)

  return result_list.difference(pastResult_list)

def line_notification(Message):
    line_notify_token = sys.argv[1]
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message = '\n' + Message 
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)

result_diff = get_result_diff(PATH + "result", PATH + "pastResult")

if (result_diff):
  line_notification('There is a new post on Code!')
  for x in result_diff:
    line_notification('https://www.kaggle.com/' + x)
