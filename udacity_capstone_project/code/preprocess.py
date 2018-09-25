from keras.models import Model
import h5py

def get_image_tensors(file_path,image_size, preprocess_input = None):
    from keras.preprocessing.image import ImageDataGenerator
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_files = file_path + 'train'
    test_files = file_path + 'test'
    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = gen.flow_from_directory(train_files, image_size, shuffle=False, batch_size=128)
    test_generator = gen.flow_from_directory(test_files, image_size, shuffle=False, batch_size=128)
    #或者使用以下代码
    #train_of_tensor = [preprocess_input(path_to_tensor(img_path, image_size[0], image_size[1])) for img_path in tqdm(train_files)]
    #返回(n_samples, target_size, 3)的4维张量
    #train_tensors = np.vstack(train_of_tensor)

    #test_of_tensor = [preprocess_input(path_to_tensor(img_path, image_size[0], image_size[1])) for img_path in tqdm(test_files)]
    #返回(n_samples, target_size, 3)的4维张量
    #test_tensors = np.vstack(test_of_tensor)

    return (train_generator, test_generator)


def get_bottleneck_features_from_ResNet50(files_path):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    (train_generator, test_generator) = get_image_tensors(files_path,(224,224),preprocess_input)
    train_data = ResNet50(weights = 'imagenet',include_top = False).predict_generator(train_generator,len(train_generator))
    test_data = ResNet50(weights = 'imagenet',include_top = False).predict_generator(test_generator,len(test_generator))
    with h5py.File("ResNet50.h5") as h:
        h.create_dataset("train", data=train_data)
        h.create_dataset("test", data=test_data)
        h.create_dataset("target", data=train_generator.classes)

def get_bottleneck_features_from_InceptionV3(files_path):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    (train_generator, test_generator) = get_image_tensors(files_path,(299,299),preprocess_input)
    train_data = InceptionV3(weights = 'imagenet',include_top = False).predict_generator(train_generator,len(train_generator))
    test_data = InceptionV3(weights = 'imagenet',include_top = False).predict_generator(test_generator,len(test_generator))
    with h5py.File("InceptionV3.h5") as h:
        h.create_dataset("train", data=train_data)
        h.create_dataset("test", data=test_data)
        h.create_dataset("target", data=train_generator.classes)

def get_bottleneck_features_from_Xception(files_path):
    from keras.applications.xception import Xception, preprocess_input
    (train_generator, test_generator) = get_image_tensors(files_path,(299,299),preprocess_input)
    train_data = Xception(weights = 'imagenet',include_top = False).predict_generator(train_generator,len(train_generator))
    test_data = Xception(weights = 'imagenet',include_top = False).predict_generator(test_generator,len(test_generator))
    with h5py.File("Xception.h5") as h:
        h.create_dataset("train", data=train_data)
        h.create_dataset("test", data=test_data)
        h.create_dataset("target", data=train_generator.classes)

def get_bottleneck_features_from_InceptionResNetV2(files_path):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
    (train_generator, test_generator) = get_image_tensors(files_path,(299,299),preprocess_input)
    train_data = InceptionResNetV2(weights = 'imagenet',include_top = False).predict_generator(train_generator,len(train_generator))
    test_data = InceptionResNetV2(weights = 'imagenet',include_top = False).predict_generator(test_generator,len(test_generator))
    with h5py.File("InceptionResNetV2.h5") as h:
        h.create_dataset("train", data=train_data)
        h.create_dataset("test", data=test_data)
        h.create_dataset("target", data=train_generator.classes)
