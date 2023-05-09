# DS482 Face Recognition - Adversial Examples as Privacy Defenses

### Team 7
- Carla Franzone
- Rivka Ziegler
- Judith Goldberg
- Eitan Greenberg
- Elay Sason
- Jonas Raedler

### File Structure

- The actual AI system can be found in the vgg_face2_transfer_learning notebook.
- The audit part can be found in the vgg_face2_adversarial_examples notebook.
- The cloaking process can be found in the cloak notebook
- extract_faces.py takes the images from the original Kaggle dataset (linked in the notebook) and extracts the face from the image. These extraced faces for all 17 celebrities are stored in the celebrity_faces_cropped_dataset folder
- The folder vgg_face_17_celebs contains the saved model after re-training
- The folder vgg_face_adversarial contains the saved model after re-training with adversarial examples