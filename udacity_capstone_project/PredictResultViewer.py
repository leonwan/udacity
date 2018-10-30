"""
# author: wanleon
# date: 2018.10.20
# 用于展示测试集中预测后的结果
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

status = ["安全驾驶", "右手打字", "右手打电话", "左手打字", "左手打电话", 
                "调收音机", "喝饮料", "拿后面的东西", "整理头发和化妆", "和其他人说话"]

def view_predict_result(model=None, show_num=10, test_dir=None, out_image_size=(299, 299), preprocess_input=None):
    
    test_show_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
    test_show_generator = test_show_datagen.flow_from_directory(test_dir, out_image_size, shuffle=False, 
                                                 batch_size=1, class_mode=None)

    plt.figure(figsize=(12, 24))
    for i, x in enumerate(test_show_generator):
        if i >= show_num:
            break
        plt.subplot(5, 5, i+1)
        preds = model.predict(x)
        preds = preds[0]

        max_idx = np.argmax(preds)
        pred = preds[max_idx]

        plt.title('c%d |%s| %.2f%%' % (max_idx , status[max_idx], pred*100))
        plt.axis('off')
        x = x.reshape((x.shape[1], x.shape[2], x.shape[3]))
        img = array_to_img(x)
        plt.imshow(img)