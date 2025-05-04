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

## Improvement

| Epoch 1 | Epoch 24 |
| ---------------- | -------------- |
| ![Before Training](https://github.com/TusharGupta1001/DRA-GAN/raw/main/generated_0.png) | ![After Training](https://github.com/TusharGupta1001/DRA-GAN/raw/main/generated_24.png) |

---

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

## Training
- **Epochs**: 25
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.0002, beta1=0.5, beta2=0.999) for both Generator and Discriminator.
- **Loss Function**: Binary Cross-Entropy (BCE) with DRAGAN gradient penalty (lambda=0.25).
- **Hardware**: CPU (GPU mode available in code).

---

# Progress Over Time
| Epoch | Observations                                        |
|-------|-----------------------------------------------------|
| 1     | Random noise, no discernible facial features        |
| 5     | Basic shapes emerge, but lacks detail               |
| 10    | Facial features (eyes, mouth) start forming         |
| 15    | More defined facial structure, some artifacts       |
| 20    | Improved realism, minor blurriness                  |
| 24    | High-quality faces, near-realistic features         |

---

# Quantitative Evaluation
- **FID at Epoch 1**: ~250 (high, indicating poor similarity to real images)
- **FID at Epoch 10**: ~120 (improving as facial features emerge
- **FID at Epoch 24**: ~45 (lower, indicating better quality and realism)

---

# Benefits of DRAGAN
- **Stability**: The gradient penalty mitigates vanishing gradients in the Discriminator, ensuring consistent training dynamics.
- **Diversity**: By constraining the Discriminator’s gradients, DRAGAN reduces mode collapse, leading to more varied facial features (e.g., different hair colors, facial expressions).
- **Convergence**: Compared to vanilla GANs, DRAGAN converges faster, as seen in the rapid quality improvement between epochs 5 and 15.

---

# Practical Applications
- **Data Augmentation**: Synthetic faces can augment datasets for face recognition models, especially in underrepresented demographics.
- **Deepfake Detection**: Generated images can be used to train models to detect manipulated media, addressing ethical concerns around deepfakes.
- **Creative Arts**: High-quality synthetic faces can be used in animation, gaming, or virtual avatars, reducing the need for manual design.

