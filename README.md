# 🎨 Fake-Vision: My DRAGAN Adventure! 🚀

Welcome to my Generative AI project! 😄  
In this repository, you'll find a fun and educational implementation of **DRAGAN** (Deep Regret Analytic GAN), where I trained a model to generate **fake human faces** using 50,000 images from the **CelebA dataset**. Let’s dive in! 🎉

---

## 🤖 What’s This All About?

Imagine two AI models playing a game:
- **Generator** 🎨: Tries to draw fake faces.
- **Discriminator** 🔍: Tries to guess if the face is real or fake.

As they keep training, the Generator gets better at fooling the Discriminator — until the fake faces start looking real!

DRAGAN helps this process stay stable and improves the **quality of generated faces**.

---

## Before and After Training

| Before Training | After Training |
| ---------------- | -------------- |
| ![Before Training]([images/before_training.jpg](https://github.com/TusharGupta1001/DRA-GAN/blob/main/generated_0.png)) | ![After Training](images/after_training.jpg) |


## 🔧 How I Built It

### Step 1: 🖼️ Dataset
- Used **50,000 celebrity faces** from the [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Resized images to **64x64** for efficient training

### Step 2: 🛠️ Model Building
- **Generator**: Turns random noise into realistic faces  
- **Discriminator**: Judges whether a face is real or fake  
- Used **PyTorch** as the deep learning framework

### Step 3: 🏋️ Training
- 25 epochs  
- Batch size = 64  
- Latent vector (z) = 100 dimensions  
- DRAGAN loss function for stability  
- Images saved every few epochs to see improvements

---

## 📈 Results Over Time

| Epoch | Result |
|-------|--------|
| 1     | 🟡 Random noise and blobs |
| 5     | 🟠 Some face-like shapes |
| 10    | 🔵 Eyes and mouth appear |
| 15    | 🟣 Faces look more human |
| 20    | 🟢 Almost realistic faces |
| 24    | ✅ Pretty convincing now! |

---
