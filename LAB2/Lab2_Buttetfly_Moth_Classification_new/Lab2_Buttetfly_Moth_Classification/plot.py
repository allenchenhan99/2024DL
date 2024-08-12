import matplotlib.pyplot as plt
import json

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
ResNet50_data = load_data('./history/ResNet50_history.json')
VGG19_data = load_data('./history/VGG19_history.json')

plt.figure(figsize=(8, 6))

plt.plot(VGG19_data['train_accuracy'], label='VGG19_train_accuracy', color='blue')
plt.plot(VGG19_data['val_accuracy'], label='VGG19_validation_accuracy', color='orange')
plt.plot(ResNet50_data['train_accuracy'], label='ResNet50_train_accuracy', color='green')
plt.plot(ResNet50_data['val_accuracy'], label='ResNet50_validation_accuracy', color='red')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy curve')
plt.legend(loc='lower right')
# plt.grid(True)
plt.show()