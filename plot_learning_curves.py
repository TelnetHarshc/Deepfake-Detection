history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stopping, reduce_lr])
model.save("real_fake_classifier.h5")
import pickle
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
