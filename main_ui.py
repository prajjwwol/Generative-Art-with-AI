import tkinter as tk
from tkinter import filedialog, Label, messagebox
from PIL import Image, ImageTk
import threading
import logging
from generative_art import create_generative_artwork

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_generative_artwork(content_path, style_path):
    try:
        logging.info("Starting style transfer...")
        result_image = create_generative_artwork(content_path, style_path)
        logging.info("Style transfer completed. Updating UI...")
        # Schedule the UI update to run on the main thread
        root.after(0, display_image, result_image)
        root.after(0, progress_label.config, {'text': 'Style transfer complete. Ready for another.'})
    except Exception as e:
        logging.error(f"Failed to generate artwork: {e}")
        root.after(0, messagebox.showerror, "Error", f"Failed to generate artwork: {e}")
        root.after(0, progress_label.config, {'text': 'Error occurred. Please try again.'})

def upload_and_generate_action():
    content_paths = filedialog.askopenfilenames(title='Select Content Images')
    if not content_paths:
        logging.info("Content image selection cancelled.")
        return
    
    style_path = filedialog.askopenfilename(title='Select Style Image')
    if not style_path:
        logging.info("Style image selection cancelled.")
        return

    progress_label.config(text='Processing... Please wait.')
    # Start the generative artwork process in a separate thread to avoid UI blocking
    threading.Thread(target=start_generative_artwork, args=(content_paths[0], style_path), daemon=True).start()

def display_image(img):
    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk  # Keep a reference

root = tk.Tk()
root.title("Generative Art Creator")

# UI Setup
generate_button = tk.Button(root, text='Generate Art', command=upload_and_generate_action)
generate_button.pack()

progress_label = tk.Label(root, text='')
progress_label.pack()

image_label = Label(root)
image_label.pack()

root.mainloop()
