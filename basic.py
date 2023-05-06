import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import pandas as pd


def build_effeintnet_b7():
    """
    Builds an effeintnet_b7 model instance
    :return: effeintnet_b7 model instance
    """
    input_layer = tf.keras.layers.Input(shape=(600, 600, 3))
    efficientnet_feature_extractor = tf.keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights='imagenet')
    x = efficientnet_feature_extractor(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    # model.summary()
    return model


def build_resnet50():
    """
    Builds a resnet50 model instance.
    :return: resnet50 model instance.
    """
    input_layer = tf.keras.layers.Input(shape=(600, 600, 3))
    resizing_layer = tf.keras.layers.Resizing(224, 224)(input_layer)
    resnet50_feature_extractor = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet')
    x = resnet50_feature_extractor(resizing_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=input_layer, outputs=x)

    # model.summary()
    return model


def compare_images(im1, im2, model):
    """
    Compares two input images using L2 norm and Cosine similarity,
    and prints the metrics.
    """
    im1 = np.expand_dims(im1, axis=0)
    im2 = np.expand_dims(im2, axis=0)
    f1 = model(im1)
    f1 = np.squeeze(f1.numpy())
    f2 = model(im2)
    f2 = np.squeeze(f2.numpy())
    l2_dist = np.linalg.norm(f1 - f2)
    cosine = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
    # print(f"L2 norm: {l2_dist}, Cosine similarity: {cosine}")
    return l2_dist, cosine


def load_im(path):
    """
    Loads an image from a given path.
    :param path: string; path to image.
    :return: Image as tensor.
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (600, 600))
    return img


def main():
    models = {'resnet 50': build_resnet50(),
              'efficientnet b7': build_effeintnet_b7()}
    ds1 = glob('paitings/**/*.jpg', recursive=True)[:30]
    ds2 = glob('paitings/**/*.jpg', recursive=True)[:30]
    for model_name, model in models.items():
        l2_scores_dict = {}
        cosine_scores_dict = {}
        for idx1, val1 in enumerate(ds1):
            im1_name = val1.split('/')[-1].replace('.jpg', '')

            # l2_scores_dict[idx1] = {}
            # cosine_scores_dict[idx1] = {}

            l2_scores_dict[im1_name] = {}
            cosine_scores_dict[im1_name] = {}

            for idx2, val2 in enumerate(ds2):
                # print(f'image 1: {val1}\nimage 2: {val2}')
                im1, im2 = load_im(val1), load_im(val2)
                im2_name = val2.split('/')[-1].replace('.jpg', '')
                l2_dist, cosine = compare_images(im1, im2, model)

                # l2_scores_dict[idx1][idx2] = l2_dist
                # cosine_scores_dict[idx1][idx2] = cosine

                l2_scores_dict[im1_name][im2_name] = l2_dist
                cosine_scores_dict[im1_name][im2_name] = cosine

        df_l2 = pd.DataFrame.from_dict(l2_scores_dict)
        df_cosine = pd.DataFrame.from_dict(cosine_scores_dict)
        print(df_l2)
        print(df_cosine)
        df_l2.to_csv(f'{model_name}.l2', index=True)
        df_cosine.to_csv(f'{model_name}.cosine.csv', index=True)


if __name__ == '__main__':
    main()
