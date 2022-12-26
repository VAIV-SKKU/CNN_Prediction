# Create data inputs
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




train_dir = "/home/ubuntu/2022_VAIV_Dataset/Output5/Kospi_and_Kosdaq/4%_01_2_5/train"
test_dir = "/home/ubuntu/2022_VAIV_Dataset/Output5/Kospi_and_Kosdaq/4%_01_2_5/train"
valid_dir = "/home/ubuntu/2022_VAIV_Dataset/Output5/Kospi_and_Kosdaq/4%_01_2_5/valid"


IMG_SIZE = (224, 224) # define image size
batch_size = 128
learning_rate =1.0e-4
dropout_rate=0.35
train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="categorical", # what type are the labels?
                                                                            batch_size=batch_size) # batch_size is 32 by default, this is generally a good number
test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                           image_size=IMG_SIZE,
                                                                           label_mode="categorical")
valid_data = tf.keras.preprocessing.image_dataset_from_directory(directory=valid_dir,
                                                                           image_size=IMG_SIZE,
                                                                           label_mode="categorical")

# 1. Create base model with tf.keras.applications
base_model = tf.keras.applications.EfficientNetB7(include_top=False, drop_connect_rate=dropout_rate)

# 2. Freeze the base model (so the pre-learned patterns remain)
base_model.trainable = False

# 3. Create inputs into the base model
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

x = base_model(inputs)

print(f"Shape after base_model: {x.shape}")


x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"After GlobalAveragePooling2D(): {x.shape}")


outputs = tf.keras.layers.Dense(2, activation="softmax", name="output_layer")(x)


model_0 = tf.keras.Model(inputs, outputs)


model_0.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=["accuracy"])

path_checkpoint = "/home/ubuntu/2022_VAIV_SeoHwan/checkpoint_01/Kosdaq_Kospi_EfficientNetB7_224_4_01_2_drop35_batch128_lre4/model-{epoch:04d}.h5"
callback_cp = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint, verbose=1, period=1)
callback_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, mode='min', verbose=1)
callback_csv = tf.keras.callbacks.CSVLogger(f'/home/ubuntu/2022_VAIV_SeoHwan/training_log_01/Kosdaq_Kospi_EfficientNetB7_224_4_01_2_drop35_batch128_lre4.csv')

history = model_0.fit(train_data,
                                 epochs=20,
                                 steps_per_epoch=len(train_data),
                                 validation_data=valid_data,
                                 callbacks=[callback_cp, callback_csv, callback_stop])


path_model = f'/home/ubuntu/2022_VAIV_SeoHwan/saved_models2/Kosdaq_Kospi_EfficientNetB7_224_4_01_2_drop35_batch128_lre4.h5'
model_0.save(path_model, overwrite=True)
