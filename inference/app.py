import json
import tensorflow as tf
import numpy as np


mnist = tf.keras.datasets.mnist
save_path = 'model/imageclass/2'
loaded = tf.saved_model.load(save_path)
infer = loaded.signatures["serving_default"]
#definig class name
class_name = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
           'shirt', 'sneaker', 'bag', 'ankle boot']


def lambda_handler(event, context):
    
    x = event['x']

    output = (infer.structured_outputs['softmax'])
    labeling = infer(tf.constant(x))[output.name]
    print(labeling)

    decoded = class_name[np.argmax(labeling)]
    print("Result after saving and loading:\n", decoded)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": decoded,
            }
        ),
    }
