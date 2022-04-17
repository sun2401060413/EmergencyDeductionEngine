import json
from urllib import request
from urllib.request import Request


def do_post(url, data):
    headers = {'Accept-Charset': 'utf-8', 'Content-Type': 'application/json'}
    params = bytes(data, 'utf8')
    req = Request(url=url, data=params, headers=headers, method='POST')
    try:
        response = request.urlopen(req)
        if(response):
            return response.read().decode()
        else:
            return response
    except Exception as e:
        return e


def predict(data, url):
    msg = json.dumps(data, ensure_ascii=False)
    retval = do_post(url, msg)
    if(retval is not None):
        return retval
    else:
        return None


def do_test(url):
    idx = ["brightness","burn","around_temp",
            "vehicle_oxygen","vehicle_status",
            "around_visibility","vehicle_temp----","person_status"]
    fname = "./test_data_{0}.json".format(idx[0])
    with open(fname, 'r', encoding='utf-8') as f:
        param = json.load(f)
        jdata = predict(param, url)
        print(jdata)
        #data = json.loads(jdata)
        #jsonString = json.dumps(data, ensure_ascii=False, indent=4)
        #print(jsonString)
        print('-'*50)


# url = "http://192.168.0.74:55574"
url = "http://localhost:55576"
if(__name__ == '__main__'):
    do_test(url)
