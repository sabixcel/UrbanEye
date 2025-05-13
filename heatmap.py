import tensorflow as tf
import numpy as np
import cv2


### function to compute grad-cam heatmap
def get_gradcam_heatmap_sequential(base_model, classifier_model, image, class_index, last_conv_layer_name):
    ### build the model that outputs the last conv layer's output and final features
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    ### record gradients
    with tf.GradientTape() as tape:
        conv_outputs, features = grad_model(tf.expand_dims(image, axis=0)) ### pass the image through grad_model
        predictions = classifier_model(features) ### then through classifier_model to get predictions
        loss = predictions[:, class_index] ### loss for the target class

    ### compute gradients of the loss based on conv layer output
    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(conv_outputs, weights), axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    return cam #output is a normalized heatmap (2D array)

### function to overlay the heatmap on top of the original image (for visualization)
def overlay_heatmap(image, heatmap, alpha=0.4, cmap='jet'):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_rgb = image.astype('uint8')
    superimposed_img = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

### function to load the image, predict its class and then generate grad-cam heatmap rendering
def generate_heatmap(image_path, class_labels, model, IMG_SIZE, LAST_CONV_LAYER):
    base_model = model.layers[0]
    classifier_head = tf.keras.Sequential(model.layers[1:])

    img_raw = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img_raw)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array.copy())

    preds = model.predict(np.expand_dims(img_preprocessed, axis=0), verbose=0)
    pred_label = np.argmax(preds[0])

    heatmap = get_gradcam_heatmap_sequential(base_model, classifier_head, img_preprocessed, pred_label, LAST_CONV_LAYER)
    superimposed = overlay_heatmap(img_array, heatmap)

    return superimposed, class_labels[pred_label], preds[0][pred_label]
