"""
This script can get the plots and details for a model.
Enter model key for the model you want to retrieve.
"""

import matplotlib.pyplot as plt
from utils import get_model_tracker

MODEL_KEY = 12
MODEL_TRACKING_FILE = "./model_tracker.pickle"


model_tracker = get_model_tracker(file=MODEL_TRACKING_FILE, folder_path=None)

# Get the details for the model in question
info = model_tracker[MODEL_KEY]
results = info[1]

plt.plot(results['loss'], label='train loss')
plt.plot(results['val_loss'], label='val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('model Loss')
plt.legend()
plt.show()
plt.savefig('Val_loss')

plt.plot(results['accuracy'], label='train accuracy')
plt.plot(results['val_accuracy'], label='val accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('model accuracy')
plt.legend()
plt.show()
plt.savefig('Val_acc')

plt.plot(results['recall'], label='train recall')
plt.plot(results['val_recall'], label='val recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.title('model recall')
plt.legend()
plt.show()
plt.savefig('Val_recall')

