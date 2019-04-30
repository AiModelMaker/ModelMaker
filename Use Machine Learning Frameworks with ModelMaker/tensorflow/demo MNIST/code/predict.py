import tensorflow as tf
import flask
import numpy as np

model_meta_data = '/opt/ai/model/model.ckpt.meta'
model_ckpt = '/opt/ai/model/model.ckpt'


flask_instance = flask.Flask(__name__)
@flask_instance.route('/ModelMaker/predict', methods=['POST'])
def predict():
    data = {"success": False}
    tmp = flask.request.get_json()['image']
    print(type(tmp))
    if flask.request.method == "POST":
        if True:
            image = np.array(tmp)
            
            print(image.shape)
            image = np.reshape(image, [1,28*28])
            r = sess.run(result, feed_dict={
                    x:image , keep_prob: 1.0})
            r = r[0]
            data["predictions"] = []
            for i in range(len(r)):
                pred = {"label": str(i), "probability":float(r[i])}
                data["predictions"].append(pred)
            
            print(data["predictions"])
                
            data["success"] = True
            
    return flask.jsonify(data)



if __name__ == '__main__':
    # Load the network and restore the checkpoint
    ######################
    # Select the network #
    ######################
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_meta_data)
        saver.restore(sess, model_ckpt)
        ########################### 
        result =  tf.get_default_graph().get_tensor_by_name('Softmax:0')
        x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
        y_ = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('Placeholder_2:0')
        flask_instance.run(host = '0.0.0.0', port = 10000)
        
