# COVID-19 Detection from Chest X-Rays
### Jordan University of Science and Technology
### Course: Deep Learning
### Team Members:

* Abdel Rahman Alsheyab
* Mohammad Alkhasawneh
* Osamh Al Shra'h
* Nidal Shahin

## Description
For the Semester-Final Team **Project**, my friends and I have built a Deep Learning model to classify chest X-ray images into **Normal**, **COVID-19**, and **Pneumonia (Viral/Bacterial)** cases. Using almost **34K CXR images of size 256x256** downloaded from **COVID-QU-Ex** Dataset on kaggle (https://www.kaggle.com/datasets/anasmohammedtahir/covidqu). Citations are mentioned in the link and the uploaded notebook.

**Why X-rays?**  They are **fast**, **non-invasive**, and **widely available**, making them an effective screening tool.

**Impact:**
* Helps hospitals prioritize treatment and isolate cases early.
* Reduces dependency on PCR tests, which can be slow and resource-intensive.
* Supports underdeveloped regions where advanced diagnostic tools are limited.

### Notes:
* We downloaded specific partitions of the original kaggle dataset and restructered it in our own way.
  * Divided into **Train (70%), Dev (15%), Test (15%)**.
  * Three **labels.csv** files were created, one for each.
  * (CNP_DS) Dataset link: (https://drive.google.com/file/d/1UUJfTyv6R1qYcePiGS6xxp66EYAfq1_H/view?usp=drive_link).
* We used multiple libraries in this project *mainly* **pytorch** for the CNN model and **matplot** for visuals.
* **You might have to change the paths to match your own files directories.**
* You will have to manually create the (.pth) path files we used in our code to save models with the best results.
* The code works on devices with or without GPUs.
* Two files where uploaded:
  * **Notebook** (.ipynb) file which includes the code and the outputs.
  * **Report** (PDF) file that covers the topic, workflow, results and challenges.
