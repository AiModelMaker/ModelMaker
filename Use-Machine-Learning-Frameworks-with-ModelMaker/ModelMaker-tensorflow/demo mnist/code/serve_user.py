import json
import tensorflow as tf


def input_fn(request):
    # data = json.loads(request_body)
    # input_data = json.dumps(request.json)
    input_data = request.get_data()
    return input_data



def output_fn(result):
    return json.dumps(result)
    
