"""云片自动短信"""
import requests, json
parameter = {"apikey":"4b83cdae270476306c65536ad40f179f",
            "mobile":"15905152631",
            "text":"【NAU Analyst】亲爱的李一繁，您的代码于2021年15月5日运行成功，共耗时2h。"}
response = requests.post("https://sms.yunpian.com/v2/sms/single_send.json",data=parameter)
print(json.loads(response.text))
