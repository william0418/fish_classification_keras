import sys
print(sys.version)

from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

batch_size = 4

epochs = 50

train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory('fish/train_data', target_size = (800, 800), batch_size = batch_size)

print("""""""""""""""""""""""""""""""""""")

print(train_generator)

print("""""""""""""""""""""""""""""""""""")

image_numbers = train_generator.samples

print (train_generator.class_indices)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory('fish/test_data', target_size = (800,800), batch_size = batch_size)



base_model = ResNet50(weights = 'imagenet', include_top = False, pooling = 'max')

predictions = Dense(4, activation='softmax')(base_model.output)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('./weights.h5')

model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=["acc"])

model.fit_generator(train_generator,steps_per_epoch = image_numbers // batch_size, epochs = epochs, validation_data = test_generator, validation_steps = batch_size)


model.save_weights('weights.h5')
