from getKerasModel import getModel, get_callbacks
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from plot3d import plotmy3d


###
# Main
###
# Preprocessing
train = pd.read_json("./data/train.json/data/processed/train.json")
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
Xband_1 = np.array([np.array(b).astype(np.float32).reshape(75, 75) for b in train["band_1"]])
Xband_2 = np.array([np.array(b).astype(np.float32).reshape(75, 75) for b in train["band_2"]])
meanX = ((Xband_1 + Xband_2) / 2)
Xtrain = np.concatenate([Xband_1[:, :, :, np.newaxis], Xband_2[:, :, :, np.newaxis], meanX[:, :, :, np.newaxis]], axis=-1)
# plotmy3d(meanX[1, :, :], 'Ship')
target_train = train['is_iceberg']
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(Xtrain, target_train, random_state=1, train_size=0.75)


# learn
file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)
gmodel = getModel()
gmodel.fit(
    X_train_cv,
    y_train_cv,
    batch_size=24,
    epochs=50,
    verbose=1,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks
)

# Show Score
gmodel.load_weights(filepath=file_path)
score = gmodel.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# make submission file
test = pd.read_json("./data/test.json/data/processed/test.json")
X_band_test_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([
    X_band_test_1[:, :, :, np.newaxis],
    X_band_test_2[:, :, :, np.newaxis],
    ((X_band_test_1 + X_band_test_2) / 2)[:, :, :, np.newaxis]],
    axis=-1
)
predicted_test = gmodel.predict_proba(X_test)
submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('sub.csv', index=False)
