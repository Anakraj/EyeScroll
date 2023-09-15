# EyeBot

Tracks eye positions and automatically scrolls webpages for hands-free reading.

For facial landmark recognition, we employ a trained convolutional neural network adapted from dlib. Our implemented algorithms differentiate anatomical parts of the eye and isolate pupil movements. Then using a buffer, range analysis, and average pixel counter we determine the optimal time to scroll down a page.  

Our backend algorithm conceptualizes future human-computer interactions with PC/mobile devices as well as heads-up displays in AR/VR headsets.

To use:
1. Ensure you meet requirements: `pip install -r requirements.txt`
2. Download files in the git repository
3. Download https://drive.google.com/file/d/1wAGwvurC9U3YbNTAdhZHpDdDj_zT6zsE/view?usp=sharing and name it shape_predictor_68_face_landmarks.dat (Eye Model Credit: antoinelame)
4. Put that file in a folder named trained_models (this folder should be at the same levels as the files downloaded from the git repository.
