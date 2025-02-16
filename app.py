from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import io
import base64
from datetime import datetime

now = datetime.now()

app = Flask(__name__,template_folder='tempelate')
app.config['UPLOAD_FOLDER'] = 'tempelate'
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])

# Load the pre-trained ResNet model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
model.eval()
with open('imagenet_classes.json', 'r') as f:
    labels = json.load(f)

# Define the allowed file extensions function
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define the prediction function
def predict(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Apply the necessary transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225        ])
    ])
    img = transform(img)

    # Make a prediction using the ResNet model
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        prediction = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 5 predicted classes and their probabilities
    top5_probs, top5_classes = torch.topk(prediction, 5)

    # Convert the class indices to class names using the ImageNet labels
    top5_names = [labels[str(c.item())] for c in top5_classes]

    # Return the top 5 predicted classes and their probabilities
    return list(zip(top5_names, top5_probs.tolist()))

# Define the Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    # Get the uploaded file from the HTML form
    file = request.files['image']

    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Save the file to the uploads folder
        try:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img.show()

            data = io.BytesIO()
            img.save(data, "JPEG")

            encoded_img_data = base64.b64encode(data.getvalue())

            # Make a prediction using the ResNet model
            prediction = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prediction = str([[i[0][1],i[1]] for i in prediction])
            encoded_img_data = base64.b64encode(data.getvalue())

        except:
            prediction = "Upload JPEG "
            encoded_img_data = 'pass'

        # # Delete the uploaded file
        # os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Return the prediction to the HTML page
        return render_template('index.html', user_image=encoded_img_data ,prediction=prediction)

    # If the file is not allowed or not uploaded, redirect to the homepage
    prediction = "Upload Image with JPEG extention"
    encoded_img_data = 'pass'
    return render_template('index.html', user_image=encoded_img_data ,prediction=prediction)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)