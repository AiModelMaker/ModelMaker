# 图像分类算法（inception_v3）
## 1.算法介绍
　　平台预置的图像分类算法，采用inception_v3模型，用户只需要将自己需要训练的数据按要求上传到s3，即可使用自己的数据训练模型。训练好模型后，只需要进行在线部署，即可获得在线推理接口。  
## 2.数据集格式要求
　　训练数据采用文件夹的方式进行组织，train、validation、test三个文件夹分别代表训练、验证、测试三个数据集，数据集中的文件夹名代表分类名称，每个分类文件中存放对应分类的数据。
　　组织方式如下：
       
           data                         //训练数据所在的文件夹
                train                      //训练集
                    class_name1            //分类1
                        image1_name.JPEG  //分类1 图片1 目前支持*.jpeg, *.png`两种格式
                        image2_name.JPEG   //分类1 图片2
                        …
                        imageN_name.PNG   //分类1 图片N
                    class_name2             //分类2
                    …
                    class_nameN            //分类N
                validation                 //验证集（可选）
                    class_name1            //分类1
                        image1_name.JPEG  //分类1 图片1 
                        image2_name.JPEG   //分类1 图片2
                        …
                        imageN_name.PNG   //分类1 图片N
                    class_name2             //分类2
                    …
                    class_nameN            //分类N
                test                       //测试集（可选）
                    class_name1            //分类1
                        image1_name.JPEG  //分类1 图片1 
                            image2_name.JPEG   //分类1 图片2
                            …
                            imageN_name.PNG   //分类1 图片N
                    class_name2             //分类2
                    …
                    class_nameN            //分类N
## 3.超参定义
　　待补充  
## 4.推理接口说明
### 请求接口：
　　采用HTTP multipart/form-data格式，将推理图片POST到推理接口进行请求。使用CURL命令发送预测请求进行测试，请求格式如下：
    
   	curl -F 'image=@图片路径' -H ' Authorization:AK值' -X POST 在线服务地址 –v
  			“-F”是指上传数据的是文件，本例中参数名为images，这个名字可以根据具体情况变化，@后面是图片的存储路径。
  			“-H”是post命令的headers，Headers的Key值为Authorization，Authorization值为发布时，获取到的AK值。
  			“POST”后面跟随的是在线服务的调用地址。
### 响应内容：
　　响应内容为json格式，返回请求图片，经过推理后，归属于每个分类的概率，概率越高，则此图片归属于此分类的可能性越高。
    
    {
    	"success":true,
        "predictions":[
            {
                "label":"class_name1",
                "probability":0.19710992276668549
            },
            {
                "label":"class_name2",
                "probability":-0.9677053689956665
            },
            {
                "label":"class_name3",
                "probability":0.9136306047439575
            }
         ]
    }
