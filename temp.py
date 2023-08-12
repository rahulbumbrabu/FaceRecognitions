import numpy as np
import cv2
from keras.models import load_model
from sklearn.preprocessing import normalize

# Load the pre-trained face recognition model
model = load_model('path_to_pretrained_model.h5')

# Process a single image and generate its embedding
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))  # Resize to model's input size
    image = image / 255.0  # Normalize pixel values
    embedding = model.predict(np.expand_dims(image, axis=0))[0]
    embedding = normalize(embedding.reshape(1, -1))  # L2 normalize
    return embedding

# Database of known members
known_members = {
    'member1': process_image('path_to_member1_image.jpg'),
    'member2': process_image('path_to_member2_image.jpg'),
    # Add more members
}

# Threshold for considering a match
similarity_threshold = 0.5

# Process a new image and identify present members
def identify_present_members(new_image_path):
    new_embedding = process_image(new_image_path)
    present_members = []
    for member, member_embedding in known_members.items():
        similarity = np.dot(new_embedding, member_embedding.T)
        if similarity > similarity_threshold:
            present_members.append(member)
    return present_members

# Example usage
new_image_path = 'path_to_new_image.jpg'
present_members = identify_present_members(new_image_path)
print("Present Members:", present_members)
