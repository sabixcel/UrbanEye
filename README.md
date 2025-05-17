# UrbanEye

## Project Overview

This project is built to **help identify problems in urban environments and suggest immediate solutions**, using computer vision and deep learning. The system allows citizens to upload images of their surroundings — such as roads, buildings, parks, or public spaces — and returns both a **predicted issue** and a **suggested response or action**.

Examples include:
- Detecting **garbage** on the street and suggesting **sanitation intervention**
- Identifying **potholes** and recommending **municipal repair**
- Spotting **graffiti** and prompting **public maintenance**
- Recognising **illegal parking** for **law enforcement**

---

![image](https://github.com/sabixcel/UrbanEye/blob/main/Demo/Screenshot%202025-05-13%20232145.png)

## Dataset

We used a combination of publicly available datasets:

1. *HydroShare Dataset*: [HydroShare – Urban Image Dataset (https://www.hydroshare.org/resource/24866122a6ee456c8f7c80aa87a9abcb/)
2. *Roboflow Community Datasets*:
   - Garbage and litter detection
   - Graffiti and vandalism
   - Road shield and sign damage
   - Pole and streetlight defects
   - Craws on electric wires
   - Cracks on buildings and walls
   - Peeling paint on public infrastructure
   - Bears and wildlife hazard detection
   - Flooded roads and areas
   - Illegal or improper parking
3. *Also some free images from*: https://unsplash.com

These datasets were preprocessed and unified into a multi-class classification task.
---

##  Model Choice: ResNet50 + Transfer Learning

We chose **ResNet50** for its balance between accuracy and computational efficiency. ResNet50 uses **residual connections** to solve the vanishing gradient problem and performs well on image classification tasks.

We applied **transfer learning** by:
- Starting with a ResNet50 model pretrained on **ImageNet**
- Replacing the final classification layers with a custom head suited for our classes
- Fine-tuning the model using our curated dataset

This approach:
- Reduces training time significantly
- Requires fewer labelled samples
- Leverages general image features learned from large-scale datasets

---

## Training Environment

All training was conducted in **Google Colab**, which provided:
- Free access to GPU resources (Tesla T4)
- Integration with Google Drive for dataset storage
- Easy-to-manage notebook-based workflows

Despite the hardware limitations of free-tier environments, we were able to **achieve good performance** and efficient convergence thanks to:
- The use of transfer learning
- A well-curated dataset
- Batch size and augmentation tuning

---

## User Interface

The project features a **simple and intuitive web interface** built with Streamlit.

- Citizens can **upload an image** (max size: **200MB**)
- The system returns:
  - **Predicted issue** (e.g.`trash`, `crack`, `illegal parking`)
  - **Prediction confidence score**
  - **Recommended solution or action** (custom mapped based on class)

This makes it easy for the general public to participate in improving their city by reporting visible issues through photos.

---

![image](https://github.com/user-attachments/assets/9ea056de-636c-4163-9bcb-07642ff27446)

** Optimizer: Adam**
**learning rate = 0.0001**



![image](https://github.com/user-attachments/assets/6cf7cfa4-508c-4de0-9e77-4cfa01e20b2a)


![image](https://github.com/user-attachments/assets/1d75844f-7450-4787-9024-5ae6e0f7f130)


![image](https://github.com/user-attachments/assets/eb6c8f3f-b42c-4438-88e3-ef6854376137)


![image](https://github.com/user-attachments/assets/eb2fd602-0ad1-48ec-8d78-5de74e3c5ebd)
    
      
##  How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/city-issue-detector.git
cd city-issue-detector

2. Download the model from google drive: https://drive.google.com/file/d/1ReL5KWwVVQyQesVYtIDe_UFULX3JvoHe/view?usp=sharing

3. Install dependencies

pip install -r requirements.txt


4. Run the App

If you want to run only on local browser: streamlit run UrbanEye.py
If you want to open the app also on the phone (connected to same network): streamlit run UrbanEye.py --server.address=0.0.0.0
Note: please change the IP address with the IPv4 ADdress from your network.

5. Access the Web Interface

The app will launch in your browser automatically.

Use it from desktop or mobile by connecting to the local network IP.


6. Upload an image or a zip archive with multiple images, with a city problem and get your solution.


Future Work
- Planned improvements for the next version:

- Cross-validation for model evaluation

- More extensive real-world testing

- Develop a dedicated mobile application

- Expand the dataset in underrepresented categories (e.g., pole defects, floods)

- Add geolocation tagging for smarter reporting

- Introduce a severity level ranking for each problem

7. Final dataset used (google drive link): https://drive.google.com/file/d/11t-BCIoe8FTalWe2sxqwBun8Iig19aOT/view?usp=sharing
