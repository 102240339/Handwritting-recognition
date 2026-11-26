# Handwritten Digit Recognition

This project classifies handwritten digits from 0 through 9 using the MNIST dataset. It includes a LeNet-style convolutional neural network, reproducible model-comparison experiments, a Flask drawing demo, and the project report.

## Project structure

- `train_digit_recognizer/train_digit_recognizer.py`: trains and saves the CNN.
- `train_digit_recognizer/run_mnist_experiments.py`: runs cross-validation, hyperparameter searches, final model comparisons, and the training-size experiment.
- `app.py`: serves the browser-based drawing demo.

## Web demo

Please visit: https://handwritting-recognition.onrender.com/

I use the free web service of Render to deploy so the web may have some delay.

## Train the CNN

The training workflow additionally needs TensorFlow or Keras and Matplotlib. Install those packages in the environment used for model development, then run:

```powershell
cd ..\Handwritting-recognition\train_digit_recognizer
python train_digit_recognizer.py
```

The script reserves 10% of the official training split for validation and evaluates the official test split once after training.

## Reproduce the experiments

The experiment workflow additionally needs TensorFlow or Keras, scikit-learn, and Matplotlib:

```powershell
cd ..\Handwritting-recognition\train_digit_recognizer
python run_mnist_experiments.py
```

The script performs cross-validation and grid search using training data, refits the selected classical estimators on the full training split, evaluates them on the official test split, and runs the training-size experiment.

## Deployment

`render.yaml` contains the Render web-service configuration. The deployed service uses the compact weights in `model_weights.npz`.
