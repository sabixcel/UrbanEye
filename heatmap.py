
import numpy as np
import tensorflow as tf
import cv2

def get_gradcam_heatmap_sequential(base_model, classifier_model, image, class_index, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, features = grad_model(tf.expand_dims(image, axis=0))
        predictions = classifier_model(features)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(conv_outputs, weights), axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    return cam

def overlay_heatmap(image, heatmap, alpha=0.4, cmap='jet'):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_rgb = image.astype('uint8')
    superimposed_img = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img
