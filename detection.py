import cv2
import numpy as np


import tensorflow as tf
model=tf.keras.models.load_model("keras_model.h5")
print(model)
video=cv2.VideoCapture(0)
while True:
    dummy,frame=video.read()
    image= cv2.resize(frame,(224,224))
    print(frame.size)
    text_image=np.array(image,dtype=np.float32)
    print(text_image.size)
    normalized_image=frame/255.0
    prediction=model.predict(normalized_image)
    print(prediction)
    cv2.imshow("result",frame)
    key=cv2.waitKey(1)
    if key ==32:
        break
video.release()
cv2.destroyAllWindows()