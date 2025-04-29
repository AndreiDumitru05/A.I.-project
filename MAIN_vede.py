import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Încărcă modelul salvat
model = tf.keras.models.load_model("mnist_model.h5")

# Dimensiuni pentru fereastra de desen (se vor scala la 28x28 pentru predicție)
canvas_width = 280
canvas_height = 280

# Setarea ferestrei principale
window = tk.Tk()
window.title("Desenează o cifră")

# Crearea canvas-ului
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Creăm o imagine PIL pentru a putea procesa desenul
image1 = Image.new("L", (canvas_width, canvas_height), 255)
draw = ImageDraw.Draw(image1)

# Funcția de desen: se desenează un cerc mic (oval) pe canvas și pe imaginea PIL
def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

# Funcție pentru a curăța canvas-ul
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    result_label.config(text="Predicted Digit: None")

# Funcție pentru predicția cifrei desenate
def predict_digit():
    # Redimensionarea imaginii la 28x28 pixeli
    img = image1.resize((28, 28))
    # Inversarea culorilor (deoarece fundalul este alb și cifra neagră)
    img = ImageOps.invert(img)
    # Conversia imaginii într-un array numpy și normalizarea valorilor
    img_array = np.array(img).astype("float32") / 255.0
    # Redimensionarea pentru a corespunde formei de intrare a modelului (batch, înălțime, lățime, canale)
    img_array = img_array.reshape(1, 28, 28, 1)
    # Realizează predicția
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    result_label.config(text=f"Predicted Digit: {digit}")

# Buton pentru predicție
predict_button = tk.Button(window, text="Predict", command=predict_digit)
predict_button.pack(pady=10)

# Buton pentru curățare
clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack(pady=10)

# Etichetă pentru afișarea rezultatului
result_label = tk.Label(window, text="Predicted Digit: None", font=("Helvetica", 16))
result_label.pack(pady=10)

window.mainloop()

#dataset mai mare 
#sa afisez vectorul cu procente de probabilitate 
#[     ]