"""
# author: wanleon
# date: 2018.10.20
# 用于切分训练集数据为训练集和验证集
"""
import numpy as np
np.random.seed(19906)
import pandas as pd
import os
import tqdm
import shutil

def split(choice_ids, train_pd_path, train_aug_pd_path, train_dir, val_dir, test_dir, origin_test_dir, saved_weights_dir="saved_weights"):
    """
       choice_ids: 选择作为验证集的司机id
       train_pd_path： 训练集的路径
       train_aug_pd_path： 数据增强过的训练集的路径
       train_dir：训练集路径
       val_dir：验证集路径
       test_dir：测试集路径
       origin_test_dir：原始的测试集路径
       saved_weights_dir: 保存模型权重的路径
    """

    print("验证集的司机ID:", choice_ids)

    drivers_pd = pd.read_csv(train_pd_path)
    imgs_pd = drivers_pd["img"]
    class_pd = drivers_pd["classname"]
    subject_pd = drivers_pd["subject"]

    # 按选择的司机ID分割训练集和验证集
    val_index = []
    for choice in choice_ids:
        val_index.extend(subject_pd[subject_pd == choice].index.tolist())

    test_mask = np.zeros(np.alen(subject_pd), dtype=np.bool)
    for val_i in val_index:
        test_mask[val_i] = True

    train_index = subject_pd[np.logical_not(test_mask)].index
    print("after split, the amount of train set:", np.alen(train_index), ", and the amount of validation set:", np.alen(val_index))

    # 读取被数据增强处理的图片
    drivers_aug_pd = pd.read_csv(train_aug_pd_path)
    imgs_aug_pd = drivers_aug_pd["img"]
    class_aug_pd = drivers_aug_pd["classname"]
    subject_aug_pd = drivers_aug_pd["subject"]

    exclude_index = []
    for choice in choice_ids:
        exclude_index.extend(subject_aug_pd[subject_aug_pd == choice].index.tolist())
    test_aug_mask = np.zeros(np.alen(subject_aug_pd), dtype=np.bool)
    for val_i in exclude_index:
        test_aug_mask[val_i] = True

    train_aug_index = subject_aug_pd[np.logical_not(test_aug_mask)].index
    print("split data from augmenters train data set : ", np.alen(train_aug_index))

    # 创建图像数据处理目录
    if not os.path.exists(saved_weights_dir):
        os.mkdir(saved_weights_dir)

    # 因为加载测试集时目录中也需要有子目录，将data/imgs/test目录软链接到data/imgs/test1/test
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        os.symlink('../test', test_dir+"/test")

    # 开始准备数据
    # 删除上次分离出的训练集和验证集文件，并重新创建目录
    rmrf_mkdir(train_dir)
    rmrf_mkdir(val_dir)

    X_train, X_val = imgs_pd[train_index], imgs_pd[val_index]
    y_train, y_val = class_pd[train_index], class_pd[val_index]

    # 链接训练集到新的目录中
    link_imgs(train_dir, X_train, y_train)

    # 链接增强训练集到新的目录中
    X_aug_train, y_aug_train = imgs_aug_pd[train_aug_index], class_aug_pd[train_aug_index]
    link_aug_imgs(train_dir, X_aug_train, y_aug_train)

    # 链接验证集到新的目录中
    link_imgs(val_dir, X_val, y_val)

    print("train data split done!")


def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def link_imgs(target_dir, X, y):
    """
        在新的训练或验证集目录中为图片创建到原位置的链接
    """
    print("link images to directory", target_dir)
    for img_name, target in zip(X, y):
        symlink_dir = os.path.join(target_dir, target)
        if not os.path.exists(symlink_dir):
            os.mkdir(symlink_dir)
        os.symlink('../../train/'+target+'/'+img_name, symlink_dir+'/'+img_name)


def link_aug_imgs(target_dir, X, y):
    print("link augmented images to directory", target_dir)
    for img_name, target in zip(X, y):
        symlink_dir = os.path.join(target_dir, target)
        if not os.path.exists(symlink_dir):
            os.mkdir(symlink_dir)
        os.symlink('../../train_aug/'+target+'/'+img_name, symlink_dir+'/'+img_name)
