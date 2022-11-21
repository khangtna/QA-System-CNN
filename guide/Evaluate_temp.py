#from keras.engine.saving import model_from_json
from tensorflow.keras.models import model_from_json
import datetime
import numpy as np
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
from data_processed import load_data_temp

print(datetime.datetime.now())
t1 = datetime.datetime.now()

model_json_file = "C:\\Users\\dieug\\Desktop\\NCKH QA\\CNN\\CNN_train_3c_relu.json"
model_h5_file = "C:\\Users\\dieug\\Desktop\\NCKH QA\\CNN\\weight\\CNN_doc_raw_train_2c-006-0.0488-1.0000.h5"
filename = []
values = []
def test_model(model_file):
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    try:
        loaded_model.load_weights(model_file)
        loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        score = loaded_model.evaluate(X_test, y_test, verbose=0)
        # print("%s: %f" % (loaded_model.metrics_names[0], score[0]))
        # print(max(values))
    finally:
        print(model_file)
    print(score[1])
    values.append(score[1])
    filename.append(model_file)


def test(path):
    FJoin = os.path.join
    files = [FJoin(path, f) for f in os.listdir(path)]

    for f in files:
        test_model(f)


def list_model(path):
    # path = "D:\\"
    FJoin = os.path.join
    files = [FJoin(path, f) for f in os.listdir(path)]

    for f in files:
        print(f)


X_train, y_train, X_test, y_test, X_val, y_val, embedding_layer = load_data_temp()
print("begin test list")
#test("D:\\Resources\\weight\\")
test("C:\\Users\\dieug\\Desktop\\NCKH QA\\CNN\\weight\\")
print("max:")
mmax = 0
mfile = ""
for index in range(len(values)):
    if(values[index] > mmax):
        mmax = values[index]
        mfile = filename[index]
print("lon nhat")
print(mfile)
print(mmax)
print("end test list")

#'''
json_file = open(model_json_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model 
loaded_model.load_weights(model_h5_file)
print("Loaded model thanh cong")

# evaluate loaded model on test data
loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=1)
print("%s: %f" % (loaded_model.metrics_names[0], score[0]))
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# predict
k = np.array(X_test)
print(X_test.shape)
y = k.reshape(804, 300)
print("Predict:")
prediction = loaded_model.predict_classes(y)
for i in range(len(prediction)):
    print("Predicted=%s" % (prediction[i]))
    # print("X=%s, Predicted=%s" % (y[i], prediction[i]))
#'''
print(datetime.datetime.now())
t2 = datetime.datetime.now()
print(t2-t1)
