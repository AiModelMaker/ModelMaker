# 使用自定义算法
　　ModelMaker支持完全由用户自定义算法镜像，自定义算法镜像需要先上传镜像至容器服务-本地镜像仓库中，在训练和推理部署时，选择自定义算法，并提供镜像地址，ModelMaker即可使用对应的镜像启动容器。  
　　需要关注信息如下：  
　　　　1.用户可以通过dokerfile指定容器的启动入口，也可以通过ModelMaker运行设置中的Command和Args指定。  
　　　　2.如果需要从s3下载数据集等，可以填写s3路径，ModelMaker会协助下载数据到/opt/ai/input/data/目录  
　　　　3.如果模型、log等数据需要持久化保存，需将其输出到/opt/ai/output/目录下，ModelMaker会将此目录下的数据同步到训练输出位置中设置的S3目录   
　　　　4.推理部署阶段，如果需要从s3下载模型，可以设置模型组件路径，ModelMaker会协助下载模型到/opt/ai/model/目录  
　　　　5.推理部署阶段，终端请求的url规则如下：  
　　　　　　http://js01infer.wangsucloud.com:10000/ModelMaker/predict  
　　　　　　需要采用http post协议，服务端口为10000端口，域名部分js01infer.wangsucloud.com可能会随不同的分区而变化  
　　　　　　请求的uri为ModelMaker/predict  
    