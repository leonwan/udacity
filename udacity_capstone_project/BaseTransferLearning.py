"""
# author: wanleon
# date: 2018.10.20
# 迁移学习的一些基础方法集合类
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing import image
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.layers.core import Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2
from keras.models import load_model
import DataSplit
from PredictResultViewer import view_predict_result
np.random.seed(201806)

class BaseTransferLearning:
    
    def __init__(self, root_path, saved_weights_dir = "saved_weights"):
        """
        初始化一些路径参数
        root_path: 数据集合的根目录
        saved_weights_dir：用于存储模型权重文件的路径
        """
        self.root_path = root_path
        self.train_dir = root_path + "imgs/train2"
        self.val_dir = root_path + "imgs/val2"
        self.test_dir = root_path + "imgs/test1"
        self.saved_weights = saved_weights_dir
        if not os.path.exists(self.saved_weights):
            os.makedirs(self.saved_weights)

        return None

    def data_split(self):
        """
        训练集数据切分为训练集及验证集。
        validationPids：需要被选做为验证集的司机id
        """
        self.drivers_pd = pd.read_csv(self.root_path + "drivers_img_nop081_list.csv")
        self.imgs_pd = self.drivers_pd["img"]
        self.class_pd = self.drivers_pd["classname"]
        self.subject_pd = self.drivers_pd["subject"]
        self.choices = ["p035", "p047"]

        DataSplit.split(choice_ids=self.choices, 
                      train_pd_path=self.root_path + "drivers_img_nop081_list.csv", 
                      train_aug_pd_path=self.root_path + "drivers_img_aug_list.csv", 
                      train_dir=self.train_dir, 
                      val_dir=self.val_dir, 
                      test_dir=self.test_dir, 
                      origin_test_dir=self.root_path + "imgs/test", 
                      saved_weights_dir=self.saved_weights)
        return self

    def init_params(self, preprocess_input, batch_size = 32, image_size=(224, 224)):
        """
        初始化参数
        preprocess_input: 模型对应的preprocess_input
        batch_size：批处理大小
        """
        self.preprocess_input = preprocess_input
        self.image_size = image_size
        self.batch_size = batch_size
        
        # 训练集图像生成器
        self.train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_input,
            rotation_range=10.,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.1,
            zoom_range=0.1,
            rescale=1./255
        )

        # 验证集图像生成器
        self.val_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_input,
            rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            class_mode='categorical')

        self.val_generator = self.val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            class_mode='categorical')
        return self

    def build_model(self, base_model, optimizer):
        """
        构建模型
        base_model: 基础模型
        optimizer：优化器
        """
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(10, activation='softmax', use_bias=False, kernel_regularizer=l2(0.01))(x)

        self.model = Model(inputs=base_model.input, outputs=predictions, name=base_model.name)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self

    def train_model(self, epochs=20):
        """
        训练模型
        epochs: 迭代次数
        """
        
        self.model_weight_path = self.saved_weights + '/' + self.model.name + '_model.h5'
        print("model name:", self.model.name, ", will save weight file:", self.model_weight_path)
        callbacks = [
            ModelCheckpoint(self.model_weight_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1, period=1),
            EarlyStopping(monitor="val_loss", verbose=1, mode="min", min_delta=0.0005, patience=3)
        ]

        self.history = self.model.fit_generator(
            self.train_generator,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=self.val_generator)
        return self

    def plt_train_progress(self):
        """
        绘图
        """
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.title('Training and validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Training and validation acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.show()
        return self

    def predict_test_set(self, preprocess_input):
        """
        预测测试集数据的结果
        preprocess_input：模型对应的preprocess_input
        """
        self.pred_model = load_model(self.model_weight_path)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
        pred_batch_size=128

        self.test_generator = self.test_datagen.flow_from_directory(self.test_dir, self.image_size, shuffle=False, batch_size=pred_batch_size, class_mode=None)

        self.test_generator.filenames[0]
        sub_df = pd.read_csv(self.root_path + "sample_submission.csv")

        y_preds = self.pred_model.predict_generator(self.test_generator, verbose=1)
        y_preds = y_preds.clip(min=0.005, max=0.995)
        print("y_pred shape {}".format(y_preds.shape))

        for i, fname in enumerate(self.test_generator.filenames):
            y_pred = y_preds[i]
            for k, c in enumerate(y_pred):
                sub_df.at[i, 'c'+str(k)] = c

        print(sub_df.head())

        sub_df.to_csv(self.root_path + 'pred_' + self.model.name + '.csv', index=None)
        print("predict done.")
        return self
        
        
    def view_predictd_result(self, num):
        """
        查看预测结果的样例
        num：需要展示的个数
        """
        view_predict_result(self.pred_model, show_num=num, test_dir=self.test_dir, out_image_size=self.image_size, preprocess_input=self.preprocess_input)
        return self
    
    def get_model(self, model_name):
        """
        根据model的名字加载model
        """
        return load_model(self.saved_weights + '/' + model_name + '_model.h5')
