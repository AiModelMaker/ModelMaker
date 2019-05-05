# 图片分类Demo (flower分类)
## 1.数据集介绍
本例采用了flower数据集，该数据集包含5种花朵，共3672张大小为200x200的花朵图片:  
　　daisy (633张)  
　　dandelion  (898张)  
　　roses  (641张)  
　　sunflowers  (699张)  
　　tulips  (799张)  
![Image text](https://raw.githubusercontent.com/AiModelMaker/ModelMaker/master/Use%20ModelMaker%20Built-in%20Algorithms/image%20class%20inception_v3/images/flowers.png)

## 2.数据集准备
　　需要下载github上data目录中的训练数据，并上传到对象存储空间。data目录中已经预先将flower数据集按8：2的比例，分隔成训练集(train文件夹)和验证集(validation文件夹)。  

| 花朵类型 | 训练集数量 | 验证集数量 |
| :------: | :------: | :------: |
| daisy | 507 | 126 |
| dandelion | 719 | 179 |
| roses | 513 | 128 |
| sunflowers | 560 | 139 |
| tulips | 640 | 159 |

## 3.创建训练任务
在AI平台点击 训练任务 -> 创建训练任务，进行训练任务的创建。创建训练任务时，需要关注以下信息：  
_ _ _
任务信息：  
　　　　名称：　填写此次训练的名称，如花朵识别  
　　　　算法来源选择： 预置算法  
　　　　预置算法选择： ImageClassification  
_ _ _
超参数配置：  
　　　　可以调整batch_size、max_number_of_steps等超参数，本例中使用默认参数即可。  
_ _ _
输入：  
　　　　初始模型： 可用于迁移学习，本例中放空  
　　　　数据集： 选择步骤2数据集准备中上传的s3路径  
_ _ _
输出：  
　　　　训练输出位置： 选择模型文件保存的位置  
_ _ _
运行配置：  
　　　　资源配置： 选择训练所使用的硬件资源 本例中使用最小配置 4核8G即可  
　　　　附加卷： 选择5G  
　　　　最长训练时间： 放空即可  
_ _ _
配置完成后，点击开始训练，即可开始训练。训练过程中可通过运行监控及训练日志，查看训练情况，训练完成后，任务状态会更新为训练完成。  

## 4.推理部署
训练完成后，可以在推理部署页面进行已训练模型的发布。  
_ _ _
模型管理->添加模型：  
　　　　名称： 填写此模型名称，如花朵识别模型  
　　　　模型版本： 填写此模型的版本，如1.0  
　　　　模型类型： 选择预置模型  
　　　　预置模型： ImageClassification  
　　　　模型组件路径： 填写第3步创建训练任务时，填写的 训练输出位置  
_ _ _
在线部署->创建服务：  
　　　　服务名称：　填写此推理服务的名称，如花朵识别服务  
　　　　模型及配置：  
　　　　　　模型：选择上面添加的模型，如花朵识别模型  
　　　　　　版本：选择模型版本，如1.0  
　　　　　　权重： 当前只有一个模型，请求均由此模型服务，权重选择 100%  
　　　　　　资源配置： 推理服务的资源配置，本例中使用最小配置 4核8G即可  
　　　　　　实例个数： 创建多少个实例进行推理服务，本例中选择 1即可  
_ _ _
部署完成后，可以在服务的详细页中的调用说明，查看 API调用接口、AK信息，后续模拟推理时，需要依赖这些信息。  
![Image text](https://raw.githubusercontent.com/AiModelMaker/ModelMaker/master/Use%20ModelMaker%20Built-in%20Algorithms/image%20class%20inception_v3/images/%E8%B0%83%E7%94%A8%E8%AF%B4%E6%98%8E.png)

## 5.模拟推理
推理服务部署完成后，即可通过API调用模拟推理请求。模拟代码如下：  
```
from skimage import io
import requests
img=io.imread(r'推理图片路径')
io.imshow(img)

files = {'image': ('推理图片名称', open('推理图片路径', 'rb'), 'image/jpeg', {})}
headers = {
            'Authorization':'调用说明中的AK值'
         }
r = requests.post('调用说明中的API调用接口地址', files=files, headers=headers) 
print(r.text)

```
运行效果如下：
![Image text](https://raw.githubusercontent.com/AiModelMaker/ModelMaker/master/Use%20ModelMaker%20Built-in%20Algorithms/image%20class%20inception_v3/images/prediect%20demo.png)









