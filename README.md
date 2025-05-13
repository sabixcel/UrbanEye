# UrbanEye

## Project Overview

This project is built to **help identify problems in urban environments and suggest immediate solutions**, using computer vision and deep learning. The system allows citizens to upload images of their surroundings — such as roads, buildings, parks, or public spaces — and returns both a **predicted issue** and a **suggested response or action**.

Examples include:
- Detecting **garbage** on the street and suggesting **sanitation intervention**
- Identifying **potholes** and recommending **municipal repair**
- Spotting **graffiti** and prompting **public maintenance**
- Recognising **illegal parking** for **law enforcement**

---

## 📂 Dataset

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

##  How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/city-issue-detector.git
cd city-issue-detector


2. Install dependencies

pip install -r requirements.txt


3. Run the App

streamlit run UrbanEye.py


4. Access the Web Interface

The app will launch in your browser automatically.

Use it from desktop or mobile by connecting to the local network IP.


5. Upload an image with a city problem and get your solution.


Future Work
- Planned improvements for the next version:

- Cross-validation for model evaluation

- More extensive real-world testing

- Develop a dedicated mobile application

- Expand the dataset in underrepresented categories (e.g., pole defects, floods)

- Add geolocation tagging for smarter reporting

- Introduce a severity level ranking for each problem
