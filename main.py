
# Machine Learning: Mnist Project
# Image Recognition
# MNIST --> recognizing handwritten digits

# Classification vs Regression
# Classification: Identify which group the object belongs in
# Regression: Prediction a value based on given data

# Brainstorming:
# Canvas to draw
# Draw bounding boxes around digit
# Predict digit using model

# GUI for MNIST prediction

from tkinter import *
import tkinter as tk
import tensorflow as tf
from tkinter import Canvas, Button, font
from tkinter import messagebox
from keras.models import load_model
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
import win32gui
import io
import imageio

model = load_model('turbo_model_1.h5')
#
def clear():
    canvas.delete("all")

def paint(event):
    # coordinates of cursor and draw from point
    x1, y1 = (event.x-1), (event.y-1)
    x2, y2 = (event.x+1), (event.y+1)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)

def handwriting_digits():
    canvas.postscript(file='digit.eps', colormode='color')
    print("Digit saved as digit.eps")
    process_and_predict()

def process_and_predict():
    image = Image.open("digit.eps")
    image = image.resize((28,28))
    image = image.convert('L')
    digit_array = np.array(image)
    digit_array = digit_array / 255.0
    digit_array = digit_array.reshape(1,28,28,1)
    prediction = model.predict(digit_array)
    prediction_digit = np.argmax(prediction)
    print("Predicted digit: " + str(prediction_digit))

    for i, prob, in enumerate(prediction.squeeze()):
        print("Class: " + str(i) + " Probability: " + str(prob))


root = tk.Tk()
root.title("Handwritten Digits")

canvas = tk.Canvas(root, width=300, height=300, bg="white")
canvas.pack(expand=tk.YES, fill=tk.BOTH)
canvas.bind("<B1-Motion>", paint)

recognize_button = tk.Button(root, text = "Recognize", command=handwriting_digits)
recognize_button.pack(side=tk.RIGHT)

clear_button = tk.Button(root, text = "Clear", command=clear)
clear_button.pack(side=tk.LEFT)



mainloop()
