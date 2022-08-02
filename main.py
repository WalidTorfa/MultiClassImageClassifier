from preprocess import image_recognition

image_object = image_recognition("train", ["cat", "dog"])
dataframe = image_object.getting_data()
train_gen, test_gen = image_object.preprocessing_data(dataframe)
model = image_object.build_model()
x = image_object.train_model(model, train_gen)
