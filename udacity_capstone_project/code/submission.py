import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

def create_submission(model_name, predictions, data_root_path):
    #从原始数据中读取副本
    df = pd.read_csv("sample_submission.csv")
    #获得按照提交文件要求顺序的文件名
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(data_root_path + 'test', (224,224), shuffle=False, batch_size=128)
    for index, fname in enumerate(test_generator.filenames):
        img_name = fname[5:]
        #第1行是文件名
        df.iat[index,0] = img_name
        #后面10行是每个类别的概论
        for pos in range(10):
            df.iat[index, pos + 1] = max(min(predictions[index][pos],1-10**(-15)),10**(-15))

    #生成可提交文件
    saved_file_name = 'submission/'+model_name + '_submission.csv'
    df.to_csv(saved_file_name, index=None)
