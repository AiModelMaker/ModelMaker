# 推理接口AK/SK签名认证算法详解  
## 1.AK/SK签名认证算法介绍  
　　推理接口使用HTTP请求头中的Authorization字段进行鉴权，鉴权通过后才提供推理服务，鉴权失败则返回http 401。  

### 1.1 从IAM获取AK、SK值：  
　　AK: Access Key Id  
　　SK: Secret Access Key  
### 1.2 签名头部说明（SignedHeaders）：  
　　SignedHeaders描述使用请求头中的哪些字段进行签名，字段间以‘;’进行分隔，不区分大小写。当前至少需要包含host、content-type、date，三个字段，且请求头中需要携带这三个头部。  
　　示例如下：  
　　　　SignedHeaders=host;content-type;date  
### 1.3 待签名的字符串（signingStr）：  
　　signingStr依据SignedHeaders中指定的头部对应的值进行生成，以‘\n’分隔。  
   ```
         POST /ModelMaker/predict HTTP/1.1
         User-Agent: curl/7.29.0
         Host: 501750.wangsu.service.com:10000
         Accept: */*
         content-type:application/json
         Date: Mon, 12 Jul 2019 09:45:44 GMT
```
　　如上http请求头，SignedHeaders=host;content-type;date时，则signingStr= "501750.wangsu.service.com:10000" + "\n" + “application/json” + "\n" + "Mon, 12 Jul 2019 09:45:44 GMT"  
### 1.4 签名（Signature）：  
　　Signature签名，对signingStr字符串使用SK先进行SHA1加密，加密后，进行BASE64转化。  
　　Signature=urlsafe_base64_encode (hmac_sha1(signingStr,SK))  

### 1.5构造Authorization头部：  
```
  Authorization需要包含以下信息：  
      WS-HMAC-SHA1（加密算法，当前只有WS-HMAC-SHA1）
      AK=‘用户AK值’
      SignedHeaders=‘需要拿来签名的头部’
      Signature=‘使用SK签名后的串，urlsafe_base64_encode (hmac_sha1(signingStr,SK))’
  示例：
  Authorization：WS-HMAC-SHA1   
                 AK=Zcg0eDmsZYK0cwmP1skyUmn9kwsmQM0HUU5,
                 SignedHeaders=host;content-type;date,
                 Signature=signature
  ```
### 1.6携带请求头发起请求：  
　　携带SignedHeaders中指定的请求头，及Authorization请求头，向推理服务的url发起请求即可。

## 2.示例代码  
```
from __future__ import print_function
import numpy as np
import base64
import requests
import json
import tensorflow as tf

import datetime
import base64
import hmac
import hashlib

def get_current_date():
    date = datetime.datetime.strftime(datetime.datetime.utcnow(), "%a, %d %b %Y %H:%M:%S GMT")
    return date

def to_sha1_base64(signingStr, secret):
    #print(signingStr)
    hmacsha1 = hmac.new(secret.encode(), signingStr.encode(), hashlib.sha1)
    return base64.b64encode(hmacsha1.digest()).decode()

def build_request_head():
    #用户的AK
    ak = 'MIl3JP6JBejL8wTALJFHnSBgkXwEpv5O1ZCR'
    #用户的SK
    sk = 'xVcz6SouAbPtHn7VHOc1HyCAsJZy61d6L4YMzWVCbQzq2qTSFrMmwxpBqeWcIDwI'

    #标识需要对哪些字段进行加密
    SignedHeaders = 'host;content-type;date'

    # 请求头字段说明 
    #  Host: 推理服务域名
    #  Content-Type: 请求内容类型 
    #  Date: 日期
    #  Authorization: 签名校验信息
    headers = {
        
                'Host': '501750.wangsu.service.com:10000',
                'Content-Type': 'application/json',
                'Date': get_current_date(),
                'Authorization': '签名校验信息'
             }
    
    #signingStr= host + \n + content-type + \n + date
    signingStr = headers['Host'] + '\n' + headers['Content-Type'] + '\n' + headers['Date']
    
    signature = to_sha1_base64(signingStr, sk)
    #Authorization = WS-HMAC-SHA1 
    #                AK=Zcg0eDmsZYK0cwmP1skyUmn9kwsmQM0HUU5, 
    #                SignedHeaders=host;content-type;date, 
    #                Signature=signature
    headers['Authorization'] = 'WS-HMAC-SHA1 '+ 'AK='+ ak +  ',SignedHeaders=' + SignedHeaders + ',Signature=' + signature
    print( headers['Authorization'])
    return headers

#准备请求数据
def prepare_data(n):
    """prepare data for inference"""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test.astype(np.float32)
    data = np.reshape(x_test[n], (1,784))
    return data, y_test[n]


def main():
    data, label = prepare_data(0)
    payload =  {"image": data.tolist()[0]}

    #向推理服务器地址，发起请求
    response = requests.post('http://501750.wangsu.service.com:10000/ModelMaker/predict',data=json.dumps(payload), headers=build_request_head())
    response.raise_for_status()
    
    print('\nGround Truth class is:', label)   
    print('\nThe complete response:\n\n', response.json(), '\n')


if __name__ == '__main__':
  main()
```