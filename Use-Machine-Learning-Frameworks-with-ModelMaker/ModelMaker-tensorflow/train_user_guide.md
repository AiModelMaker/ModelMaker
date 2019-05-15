# 使用 ModelMaker-tensorflow 训练模型

ModelMaker tensorflow训练部分基于 [TensorFlow Estimator API](https://tensorflow.google.cn/guide/estimators)，一种可极大简化机器学习编程的高阶 API。跟随本文档，您可以快速了解如何使用 ModelMaker 构建一个基于 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/) 的训练任务，并最终输出可用于部署的 SavedModel 模型。如果您已经熟悉 Estimator，您会发现将您的代码迁移至 ModelMaker 仅需极小的改动。



## 创建训练代码

ModelMaker 已经帮助您集成了大多数除了构建模型以外的任务，让您只需专注于模型与数据，在训练代码中，ModelMaker 大体上使用了 Estimator 所规范的形式（参见 [TensorFlow - 创建自定义 Estimator](https://tensorflow.google.cn/guide/custom_estimators)），有所不同的是，在 ModelMkaer 中，我们只需要构建四个函数：

- `model_fn`：模型函数
- `train_input_fn`：训练集输入函数
- `eval_input_fn`：验证集输入函数
- `serving_input_receiver_fn`：定义最终输出的 SavedModel 在被部署时接收推理请求的输入函数

ModelMaker 会帮助您完成其他任务，包括根据配置选择启用分布式训练、读写 checkpoints，或输出可用于部署的 SavedModel 模型等。



### 编写输入函数

您需要为训练（train）与验证（eval）分别创建命名如下的输入函数：

- `train_input_fn(path, params)`
- `eval_input_fn(path, params)`

其中，`path` 是一个字符串类型的参数，参数值是您 [上传数据集](#上传数据集) 的路径，`params` 是一个字典，从 `parameters.json` 中读取的数据和 [默认训练参数](#修改默认训练参数) 将会被传递到这里。

以下展示了一个简单的实现，要了解更多，可参阅 [预创建的 Estimator](https://tensorflow.google.cn/guide/premade_estimators#create_input_functions) ；请注意，此处只定义了 `train_input_fn`，您还需要另外定义验证集输入函数 `eval_input_fn`。

```python
# 输入函数的函数名必须命名为 train_input_fn 与 eval_input_fn
def train_input_fn(path, params):
    # 调用 TensorFlow 内建函数下载并读取 MNIST 数据集
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()
    # 将数据归一化(0-1 normalization)并转型为 float32
    x_train = tf.cast((x_train / 255.0), tf.float32)
    
	# 从 MNIST 数据集建立 dataset 实例
    # 注意，这里的 'images' 与特征列一节中的 key='images' 相对应
    dataset = tf.data.Dataset.from_tensor_slices(({'images': x_train}, y_train))
    
    # 对 dataset 进行处理：shuffle, batch & repeat
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(64)
    dataset = dataset.repeat()
    
    # 在使用分布式训练时，必须返回一个 dataset 实例
    # 在非分布式训练时，可返回含有(features, labels)的元组
    return dataset
```



### 编写模型函数

模型函数具有以下定义格式：

```python
# 函数名只能被定义为 model_fn
def model_fn(
    features,	# 从 input_fn 输入函数返回的样本特征
    labels,		# 从 input_fn 输入函数返回的样本标签
    mode,		# tf.estimator.ModeKeys 的实例，指定模型运行模式（train, eval, predict)
    params		# 额外的参数配置，与输入函数中 params 的值完全相同
):
```



一个完整的神经网络模型函数通常需要以下内容：

- [定义模型](#定义模型)
  - [定义特征列](#特征列)
  - [输入层](#输入层)
  - [一个或多个隐藏层](#隐藏层)
  - [输出层](#输出层)
- 为[三种模式](#实现预测、验证与训练)添加计算并返回 [EstimatorSpec](https://tensorflow.google.cn/api_docs/python/tf/estimator/EstimatorSpec) 的一个实例
  - [预测](#预测)
  - [验证](#验证)
  - [训练](#训练)



### 定义模型

#### 特征列

特征列是原始数据和模型输入层之间的媒介，提供了多种将输入数据（由 `input_fn` 返回）处理为特定格式的方法（详见 [特征列](https://tensorflow.google.cn/guide/feature_columns) 与 [预创建的 Estimator](https://tensorflow.google.cn/guide/premade_estimators#define_the_feature_columns) ) ，此处我们为 MNIST 数据创建一个简单的 `numeric_column`，表示将输入特征的值（28x28的矩阵）直接用作模型的输入:

```python
# 特征列定义了如何使用输入数据
# 注意，此处的 key='images' 与输入函数中的 'images' 相对应
feature_columns = tf.feature_column.numeric_column(key='images', shape=[28, 28])
```



#### 输入层

调用 `tf.feature_column.input_layer`，从输入的特征字典 `features` 和特征列 `feature_columns` 上创建模型的输入层：

```python
# 调用 input_layer 创建输入层
net = tf.feature_column.input_layer(features, feature_columns)
```



#### 隐藏层

一个（深度）神经网络模型通常含有一个或多个隐藏层，隐藏层的结构直接影响了网络的性能，此处，我们为网络添加了两个带有 dropout 的全连接层，每层有 128 个节点，使用 relu 函数作为激活函数：

```python
# 两个带有 Dropout 的全连接层
for _ in range(2):
	net = tf.layers.dropout(
            tf.layers.dense(net, 128, activation=tf.nn.relu),
            rate=0.2, training=(mode == tf.estimator.ModeKeys.TRAIN)
    )
```



#### 输出层

最后我们创建输出层，仍使用全连接层，但不使用 dropout 和激活函数，每个节点的值代表了输入数据属于某一类标签的可能性大小（但总和并不为1，要获得总和为1的结果，需要通过 softmax 等函数进行后处理）。

```python
# 创建输出层，10 个节点对应 MNIST 数据集共有十类标签（0-9）
logits = tf.layers.dense(net, 10, activation=None)
```



### 实现预测、验证与训练

在 ModelMaker 中，后台使用了 [`train_and_evaluate`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/estimator/train_and_evaluate?hl=en) 函数来帮助您进行训练与验证，并将训练好的模型输出为 SavedModel 模型，在每个不同操作中，ModelMaker 会调用模型函数 `model_fn` 并将 mode 参数设定为如下所示的值：

|   操作   |            mode 值            |
| :------: | :---------------------------: |
|   训练   |  tf.estimator.ModeKeys.TRAIN  |
|   验证   |  tf.estimator.ModeKeys.EVAL   |
| 输出模型 | tf.estimator.ModeKeys.PREDICT |

对于每一种 mode 的值，模型函数都需要返回一个 [`tf.estimator.EstimatorSpec`](https://tensorflow.google.cn/api_docs/python/tf/estimator/EstimatorSpec) 实例，除了必须返回的 `mode` 以外，不同模式下还必须给出相应的必要参数：

- 对于 `mode == TRAIN`，必须给出 `loss` 和 `train_op`
- 对于 `mode == EVAL`，必须给出 `loss`
- 对于 `mode == PREDICT`，必须给出 `predictions`



#### 预测

预测模式决定了模型在执行推理时的输出，在 ModelMaker 中，预测模式只在模型被部署时使用，一个简单的示例如下所示：

```python
# 使用 argmax 判断输出数据的标签
predicted_classes = tf.argmax(logits, 1)

# 如果处于预测模式，则返回一个带有 predictions 的 EstimatorSpec 实例
if mode == tf.estimator.ModeKeys.PREDICT:        
	return tf.estimator.EstimatorSpec(
        mode, 
        predictions={'class_id': predicted_classes},
    )
```

在特殊情况下，我们希望模型在部署时可以提供多种不同的输出，我们可以通过配置 `export_outputs` 参数来实现一个多头模型，以下是一个示例，您也可以参阅 [保存和恢复模型](https://tensorflow.google.cn/guide/saved_model#using_savedmodel_with_estimators) 以获得详细说明：

```python
# 使用 argmax 判断输出数据的标签
predicted_classes = tf.argmax(logits, 1)

# 如果处于预测模式，则返回一个带有 predictions 和 export_outputs 的 EstimatorSpec 实例
if mode == tf.estimator.ModeKeys.PREDICT:
	
    # 此处我们构建一个含有更多信息的 predictions 字典
    predictions = {
            'class_id': predicted_classes,
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
    }

    # 创建 export_outputs, 包含一个默认模式和一个只输出预测标签的模式
    # 此时，EstimatorSpec 的 predictions 参数不再具有实际意义，因为 ModelMaker平台
    # 并不调用 estimator.predict() 方法
    # 在多头模式下，必须有一个键为 DEFAULT_SERVING_SIGNATURE_DEF_KEY
    # 部署模型后，在进行推理时，默认模式对应的键为 `serving_default`
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(predictions),
        'predict_images':
        	tf.estimator.export.PredictOutput({'scores': tf.nn.softmax(logits)})
    }
        
	return tf.estimator.EstimatorSpec(
        mode, 
        predictions=predictions,
        export_outputs=export_outputs,
    )
```

请注意，在最开始的简要示例中，我们没有配置 `export_outputs`，但事实上 `export_outputs` 默认被 Estimator 根据 `predictions` 的值配置为了 `PredictOutput` 模式。



#### 验证

在训练的过程中，验证模式根据训练参数的配置每隔一段时间被调用一次（详见 [修改默认训练参数](#修改默认训练参数) ），验证部分的代码示例如下所示：

```python
# 计算损失，该损失同样会被用于训练模式
onehot_labels=tf.one_hot(labels, 10, 1, 0)
loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

# 创建精度指标
# 注意，这一参数是可选的，但我们通常会返回至少一个指标
accuracy = tf.metrics.accuracy(
    labels=labels,
    predictions=predicted_classes,
    name='acc_op'
)
    
metrics = {'accuracy': accuracy}

if mode == tf.estimator.ModeKeys.EVAL:
	return tf.estimator.EstimatorSpec(
        mode, 
        loss=loss,
        eval_metric_ops=metrics
    )
```



#### 训练

ModelMaker 使用 [`train_and_evaluate`](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/estimator/train_and_evaluate?hl=en) 对模型进行训练，

```python
# 训练操作通常被放在最后，因此使用 assert 判断
assert mode == tf.estimator.ModeKeys.TRAIN

# 创建优化器并调用优化器的 minimize 方法来创建训练操作 train_op
global_step = tf.train.get_global_step()
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss, global_step=global_step)

# 最后返回带有 loss 和 train_op 的 EstimatorSpec 实例
return tf.estimator.EstimatorSpec(
    mode,
    loss=loss,
    train_op=train_op,
)
```



至此，您已经完整编写了模型函数，最后，我们还需要为推理请求准备一个输入函数。



### 为推理请求准备输入函数

就像在训练与验证期间我们需要为模型提供输入函数一样，在模型被部署并提供服务的期间，我们也需要定义一个函数，`serving_input_receiver_fn`， 来接受推理请求，它具有以下作用：

- 为输出的 SavedModel 模型添加占位符
- 完成从收到的推理请求到模型所预期的数据格式间所需的预处理操作

该函数需要返回以下对象/函数中的一种：

- [tf.estimator.export.ServingInputReceiver](https://tensorflow.google.cn/api_docs/python/tf/estimator/export/ServingInputReceiver)
- [tf.estimator.export.TensorServingInputReceiver](https://tensorflow.google.cn/api_docs/python/tf/estimator/export/TensorServingInputReceiver)

- [tf.estimator.export.build_parsing_serving_input_receiver_fn](https://tensorflow.google.cn/api_docs/python/tf/estimator/export/build_parsing_serving_input_receiver_fn)

- [tf.estimator.export.build_raw_serving_input_receiver_fn](https://tensorflow.google.cn/api_docs/python/tf/estimator/export/build_raw_serving_input_receiver_fn)

前两个对象，`ServingInputReceiver` 与 `TensorServingInputReceiver` 都表示将接收 `receiver_tensors` 并转换成模型所期望的输入 `features`，不同的是 `ServingInputReceiver` 的返回值是一个字符串映射至 `Tensor` 或 `SparseTensor` 的字典，而 `TensorServingInputReceiver` 返回的是一个 `Tensor` 或 `SparseTensor`；

后两个函数中，`build_parsing_serving_input_receiver_fn` 根据所给出的 `feature_spec` 直接将输入的 `tf.Example` 进行解析后传递给模型，适用于推理请求以 `tf.Example` 形式到达并且无需解析以外的额外处理的情况；而最后一个函数 `build_raw_serving_input_receiver_fn` 适用于对推理请求不做任何处理，直接传递给模型的情况；

您同样可以参阅每个对象/函数的链接或者 [保存于恢复模型](https://tensorflow.google.cn/guide/saved_model#using_savedmodel_with_estimators) 获得更多信息。

在我们 MNIST 的例子中，我们使用了最后一种方式，即直接将推理请求传递给模型，这也是最简单的一种实现方式：

```python
# 与前面一样，函数名必须被命名为 serving_input_receiver_fn
def serving_input_receiver_fn():
    
    # 创建 placeholder 
    feature = tf.placeholder(tf.float32, shape=[1, 28, 28], name='images')
    
    # 将 feature 封装成字典形式， 这里的 'images' 与输入函数及特征列中的 'images' 相对应
    feature_dict = {'images': features}
    
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_dict)
```



现在，我们可以保存我们的训练代码，并将训练代码上传到 ModelMaker 平台并训练我们的模型。



## 上传数据集

在 MNIST 的例子中，由于我们直接调用了 TensorFlow 的内建函数来获得数据集，所以跳过了准备与上传数据集的步骤，但在一般的场景中，数据集往往非常庞大，需要提前进行准备。您可以将您的数据集上传到我们的 s3 存储，通过上传数据集，您的数据将会被妥善存放在我们的平台，数据存放的路径将通过 `PATH` 参数传递给 `train_input_fn` 与 `eval_input_fn` ，您可以通过任何您希望的方式来使用这些数据。



## 修改参数

通过在创建训练界面->运行设置中添加运行参数，用户可以

- 修改系统默认的训练参数
- 创建任意的自定义参数，这些参数会以字典的形式被完整地传参至 `model_fn`， `train_input_fn` 以及 `eval_input_fn` 的 `params` 参数中

### 修改默认训练参数

在创建训练界面->运行设置中添加运行参数，对以下参数赋值，可以修改系统默认的训练参数：

- `save_checkpoints_steps`: 每隔多少步存储一次 checkpoint，不能与`save_checkpoints_secs` 同时设置；默认值：None
- `save_checkpoints_secs`：每隔多少秒存储一次 checkpoint，不能与 `save_checkpoints_steps`同时设置，若两者均未设置，则该项默认为600秒，如果两项都为None，则 checkpoint 被禁用；默认值：600
- `keep_checkpoint_max`：checkpoint 的最大保存数量，达到上限后，新的 checkpoint 会替换最旧的 checkpoint，如果设置为 0 或者 None，则所有 checkpoints 都会被保存；默认值：5
- `log_step_count_steps`：每多少步显示一次 log，log包含 loss 和 global step；默认值：100

- `train_max_steps`：训练总步数，即单卡训练步数 x 单机卡数 x 机器数，若设置为 None 会一直训练下去（直到触发 OutofRange 错误（输入的训练集跑完））；默认值：10000

- `eval_steps`：每次验证的执行步数，即每次验证操作会验证的 batch 数量，如果设置为 None，会跑完整个输入验证集；默认值：100
- `eval_start_delay_secs`：在开始第一次验证前等待的秒数；默认值：120
- `eval_throttle_secs`：在重新开始新一轮验证前，需要等待的最短秒数（需要有一个新的checkpoint才能开始新一轮验证，所以是最短秒数）；默认值：600