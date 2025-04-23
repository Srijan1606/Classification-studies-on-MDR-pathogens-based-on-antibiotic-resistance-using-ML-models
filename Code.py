# Classification-studies-on-MDR-pathogens-based-on-antibiotic-resistance-using-ML-models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt

# Load data
X = pd.read_csv("snp_data.csv", header=None).values
labels = pd.read_csv("labels.csv", header=None).values

y_binary = labels[:, 0]
y_multiclass = labels[:, 1] - 1
y_multiclass_onehot = to_categorical(y_multiclass, num_classes=16)

X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
    X, y_binary, y_multiclass_onehot, test_size=0.2, random_state=42
)

# Compute class weights
binary_class_weights = compute_class_weight('balanced', classes=np.unique(yb_train), y=yb_train)
binary_class_weights_dict = {i: weight for i, weight in enumerate(binary_class_weights)}

y_multiclass_int = np.argmax(ym_train, axis=1)
multiclass_weights = compute_class_weight('balanced', classes=np.unique(y_multiclass_int), y=y_multiclass_int)
multiclass_class_weights_dict = {i: weight for i, weight in enumerate(multiclass_weights)}

def multi_output_class_weight(y_binary, y_multiclass_int):
    sample_weights = []
    for b, m in zip(y_binary, y_multiclass_int):
        w_binary = binary_class_weights_dict[b]
        w_multi = multiclass_class_weights_dict[m]
        sample_weights.append((w_binary, w_multi))
    return np.array(sample_weights).T

sample_weights_binary, sample_weights_multiclass = multi_output_class_weight(yb_train, y_multiclass_int)

# --------------------------
# Build Model with Branches
# --------------------------
def build_model(hp):
    input_dim = X.shape[1]
    inputs = Input(shape=(input_dim,), name="input")

    # Shared base
    shared = Dense(units=hp.Int("shared_units", 256, 512, step=64), activation='relu')(inputs)
    shared = Dense(units=hp.Int("shared_units_2", 128, 256, step=64), activation='relu')(shared)

    # Binary branch
    binary_branch = Dense(units=hp.Int("bin_units", 64, 128, step=32), activation='relu')(shared)
    binary_output = Dense(1, activation='sigmoid', name='binary')(binary_branch)

    # Multiclass branch
    multi_branch = Dense(units=hp.Int("multi_units", 64, 128, step=32), activation='relu')(shared)
    multiclass_output = Dense(16, activation='softmax', name='multiclass')(multi_branch)

    model = Model(inputs=inputs, outputs=[binary_output, multiclass_output])
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("lr", [1e-3, 5e-4, 1e-4])),
        loss={'binary': 'binary_crossentropy', 'multiclass': 'categorical_crossentropy'},
        metrics={'binary': 'accuracy', 'multiclass': 'accuracy'}
    )
    return model

# --------------------------
# Keras Tuner Setup
# --------------------------
tuner = kt.RandomSearch(
    build_model,
    objective='val_binary_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='snp_dual_output_branch'
)

# --------------------------
# Train
# --------------------------
tuner.search(
    X_train,
    {'binary': yb_train, 'multiclass': ym_train},
    validation_data=(X_test, {'binary': yb_test, 'multiclass': ym_test}),
    epochs=10,
    batch_size=32,
    sample_weight={'binary': sample_weights_binary, 'multiclass': sample_weights_multiclass},
    verbose=2
)

# --------------------------
# Evaluate / Predict
# --------------------------
best_model = tuner.get_best_models(1)[0]
X_new = X_test[:1]
binary_pred, multiclass_pred = best_model.predict(X_new)

pred_binary_class = int(binary_pred[0] > 0.5)
pred_class_index = multiclass_pred[0].argmax()
pred_4bit_code = format(pred_class_index, '04b')

print("Predicted species (binary):", pred_binary_class)
print("Predicted drug label index:", pred_class_index + 1)
print("Predicted 4-bit binary code:", pred_4bit_code)
