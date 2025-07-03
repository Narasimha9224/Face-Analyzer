# Face-Analyzer
Face Analyzer is a deep learning-based web application that detects and verifies whether a face in the uploaded image matches any face in the dataset. It is built using OpenCV, face recognition libraries, and a simple HTML/CSS/JS frontend.
# Web Interface
The interface provides a clean and user-friendly experience:

Upload Section:
Users can drag & drop or select an image file to upload.

Purpose:
The uploaded image is analyzed and compared with pre-trained face data to determine if there's a match.

Response Details:
After uploading, the result (match found or no match found) will be displayed in the "Response Details" section.
# How It Works
* The user uploads an image.
* The backend processes it using a face recognition model.
* The result is returned to the frontend with details on whether the face is in the dataset.
# Technologies Used
 * Python
 * Flask
 * OpenCV
 * face_recognition (dlib-based)
  * HTML, CSS, JavaScript
![Screenshot (97)](https://github.com/user-attachments/assets/ac11c10e-df8f-4344-867a-ec5abc7f9102)
# Face Matching in Action
* Below is an example of the Face Analyzer web interface in use:
 *A user uploads a photo using the Drag & Drop or Select Image option.
 *The system provides three action buttons:
* Check for Match – to compare the uploaded image with the existing dataset.
*Add to Dataset – to store a new face in the dataset for future reference.
* Reset – to clear the current input and result.
* Once the image is submitted and found in the dataset, the result is displayed:

✅ Match Found: A success message confirms that the uploaded face matches a face already stored in the dataset.
* JSON-like Response Details give structured data output.
![Screenshot (98)](https://github.com/user-attachments/assets/5a97323a-6539-4909-aa5f-f99ad0bc6e97)
![Screenshot (99)](https://github.com/user-attachments/assets/2208b682-4b3d-4b0e-8730-0dd8094895dc)




