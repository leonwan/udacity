import h5py
from keras.models import *
from keras.layers import *
np.random.seed(5000)
from keras import optimizers

def train_ResNet50(files_path):
    
    from keras.preprocessing.image import ImageDataGenerator
    from PIL import ImageFile
    from keras.applications.resnet50 import ResNet50, preprocess_input
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_files = files_path + 'train'
    image_size = (224,224)
    gen = ImageDataGenerator(preprocessing_function=preprocess_input, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = gen.flow_from_directory(train_files, image_size, shuffle=False, batch_size=128)
        
    base_model = ResNet50(weights = 'imagenet',include_top = False)
    
    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)

    # 添加一个分类器
    predictions = Dense(10, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

#     # 首先，我们只训练顶部的几层（随机初始化的层）
#     # 锁住所有 InceptionV3 的卷积层
#     for layer in base_model.layers:
#         layer.trainable = False

#     # 编译模型（一定要在锁层以后操作）
#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#     # 在新的数据集上训练几代
#     model.fit_generator(generator=train_generator, epochs=10)
    
#     # 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
#     # 我们会锁住底下的几层，然后训练其余的顶层。

    # 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # 我们选择训练最上面的两个 Inception block
    # 也就是说锁住前面249层，然后放开之后的层。
    for layer in model.layers[:164]:
       layer.trainable = False
    for layer in model.layers[164:]:
       layer.trainable = True

    # 我们需要重新编译模型，才能使上面的修改生效
    # 让我们设置一个很低的学习率，使用 SGD 来微调
    from keras.optimizers import SGD
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 我们继续训练模型，这次我们训练最后两个 Inception block
    # 和两个全连接层
    model.fit_generator(generator=train_generator, epochs=50)
    
    return model


def train_InceptionV3(files_path):
    
    from keras.preprocessing.image import ImageDataGenerator
    from PIL import ImageFile
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_files = files_path + 'train'
    image_size = (224,224)
    gen = ImageDataGenerator(preprocessing_function=preprocess_input, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = gen.flow_from_directory(train_files, image_size, shuffle=False, batch_size=128)
        
    base_model = InceptionV3(weights = 'imagenet',include_top = False)
    
    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)

    # 添加一个分类器
    predictions = Dense(10, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

#     # 首先，我们只训练顶部的几层（随机初始化的层）
#     # 锁住所有 InceptionV3 的卷积层
#     for layer in base_model.layers:
#         layer.trainable = False

#     # 编译模型（一定要在锁层以后操作）
#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#     # 在新的数据集上训练几代
#     model.fit_generator(generator=train_generator, epochs=10)
    
#     # 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
#     # 我们会锁住底下的几层，然后训练其余的顶层。

    # 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # 我们选择训练最上面的两个 Inception block
    # 也就是说锁住前面249层，然后放开之后的层。
    for layer in model.layers[:150]:
       layer.trainable = False
    for layer in model.layers[150:]:
       layer.trainable = True

    # 我们需要重新编译模型，才能使上面的修改生效
    # 让我们设置一个很低的学习率，使用 SGD 来微调
    from keras.optimizers import SGD
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 我们继续训练模型，这次我们训练最后两个 Inception block
    # 和两个全连接层
    model.fit_generator(generator=train_generator, epochs=70)
    
    return model