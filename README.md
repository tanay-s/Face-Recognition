# Face-Recognition
Training data: **face_images.mat** <br />
Training labels: There are 13 people, each with 70 images each. The images are ordered by person (70 images of person A, followed by 70 images of person B, etc.)<br />
Test data: **unknown_faces.mat** <br />
Test labels: **unknown_labels.mat** <br />
<br />
Steps followed: <br />
PCA is implemented for feature reduction. <br />
K-Means clustering is done to obtain 13 clusters. <br />
Test images are then identified using clustering or projecting on the feature vector to determine faces. <br />
