# 使用 ModelMaker 一键部署模型

ModelMaker 部署部分基于 [TensorFlow Serving](https://github.com/tensorflow/serving)，一个 TensorFlow 官方的开箱即用的部署工具，借助 TensorFlow Serving，您几乎不需要任何额外的操作就可以轻松地将训练好的模型部署至我们的服务器上。如果您还没有可用于部署的 [SavedModel](https://tensorflow.google.cn/guide/saved_model#build_and_load_a_savedmodel) 模型，请参阅 [使用 ModelMaker 训练模型] 一节获得相关的指南。



## 一键部署模型

在您将训练好的模型上传至 ModelMaker 平台的 s3 存储之后，您只需在部署模型界面选择您模型所在的位置即可一键部署，请注意，选择的路径必须是模型文件（saved_model.pb 文件 / variables 文件夹 等）所在文件夹的母文件夹，通常，一个正确的路径下的文件夹结构如下所示：

```
- /path/to/model/folder/
  	|
  	| - 1554889458/
  		| - saved_model.pb
        | - variables/
  	| - 155487837
  		| - saved_model.pb
  		| - variables/
```

其中，1554889458 等数字是由 estimator 根据时间戳自动生成的文件夹。



在您选定正确的文件夹并点击开始部署后，ModelMaker 会自动将该文件夹下最新版本（数字最大）的模型部署至服务器上，您可以通过 http 请求来进行推理



## 发送推理请求

在您部署模型后，***系统会提供接受推理请求的接口 url***，您可以通过 REST API 发送推理请求，请求的格式请参照 [TensorFlow Serving 推理请求格式](https://www.tensorflow.org/tfx/serving/api_rest#request_format_2) , 请注意，ModelMaker 只接受 Predict 格式的推理请求。



## 自定义 预处理/后处理 函数

如果您希望在服务端添加预处理或者后处理的代码，您可以通过在部署模型时上传您的数据处理代码，它包括两个部分：

- 预处理代码 `input_fn`
- 后处理代码 `output_fn`

如果您提供了自定义的 预处理/后处理 代码，您客户端所上传的 HTTP 请求的内容将不会被自动传递至 TensorFlow Serving 进行推理，而是会以 [flask request](http://flask.pocoo.org/docs/1.0/api/#flask.Request) 对象的形式传参至您的预处理代码 `input_fn`，您需要返回一个符合[TensorFlow Serving 推理请求格式](https://www.tensorflow.org/tfx/serving/api_rest#request_format_2) 的值（与 [发送推理请求](#发送推理请求) 一节一致）：

```python
# 与训练部分类似，函数名必须被命名为 input_fn
def input_fn(request):
    request = request.get_data()
    return request_body
```



后处理代码会接收 TensorFlow Serving 所返回的推理结果，您可以根据您的需求对推理结果进行后处理操作，您的返回值将会被作为 body 返回至您的客户端:

```python
# 同样，函数名是固定的
def output_fn(result):
    response_body = json.dumps(result)
    return response_body
```

