# Handwriting-Generation-Model
This Project aims to Generate Handwriting using Machine Learning methods. We have LSTM and RNN concept.

📝 **HANDWRITTEN TEXT GENERATION**  
==================================  

🚀 **PROJECT OVERVIEW**  
This pipeline takes any input text and turns it into realistic, human-like handwriting using deep learning.  

• **Data Preparation**: Splitting & normalizing the DeepWriting Dataset (https://ait.ethz.ch/deepwriting)  
• **Model Architectures**:  
  – MDN-LSTM (Graves-style) for sequence-to-sequence stroke synthesis  
  – SketchRNN-Style VAE for richer latent-space sampling  
• **Training & Evaluation**:  
  – Teacher-forcing training with MDN negative-log-likelihood, BCE (pen flag) and KL losses  
  – Validation metrics: pen-lift accuracy & average NLL  
• **Inference**:  
  – Jupyter notebook & CLI for interactive demos  
  – Scripted generation of handwriting images  

---  

📂 **DIRECTORY STRUCTURE**  
project_root/  
  data/                  Raw & split DeepWriting (.npz → per-sample .npz)  
  models/                Saved checkpoints (best.pth, full_model.pth)  
  notebooks/             Jupyter notebooks for exploration  
  src/                   Core modules  
    data.py              Dataset & collate_fn  
    model.py             TextEncoder, Decoder, SketchRNN  
    loss.py              MDN loss, VAE+KL loss  
    train.py             Training loop & checkpointing  
    generate.py          Sampling & CLI code  
  README.txt             This file  

---  

🗃️ **DATA PREPARATION**  
1. Download DeepWriting:  
   • deepwriting_training.npz  
   • deepwriting_validation.npz  
2. Split into one .npz per sample (scripts/split_deepwriting.py)  
3. Normalize strokes using provided mean & std from the monolithic .npz  

---  

🏗️ **MODEL ARCHITECTURES**  

1. **MDN-LSTM (Alex Graves 2013)**  
   – **Encoder**: Embedding → Bidirectional LSTM → summed forward/backward outputs  
   – **Decoder**: Attention over encoder outputs + autoregressive LSTM on [Δx, Δy, pen_up] + context  
   – **Mixture-Density Output**:  
     • M Gaussians → (μₓ, μᵧ, σₓ, σᵧ, ρ, π)  
     • + Bernoulli logit for pen-lift  
   – **Loss**: MDN negative-log-likelihood + binary cross-entropy for pen flag  

2. **SketchRNN-Style VAE**  
   – **Encoder**: Bidirectional LSTM → latent z (μ, log σ²)  
   – **Reparameterize**: z ∼ N(μ, σ²)  
   – **Decoder**: LSTM conditioned on z at each step → MDN + categorical pen-state output  
   – **Loss**: MDN NLL + categorical CE + KL(q(z|x) ‖ p(z))  

---  

🏋️‍♀️ **TRAINING**  
• Script: src/train.py  
• Features: gradient clipping (‖g‖ ≤ 5), LR scheduling, validation loop, best-model checkpointing  
• Metrics: Pen-up accuracy, average MDN NLL  

---  

🔍 **EVALUATION**  
Run:  
  python src/eval.py --model models/best.pth --data_dir data/deepwriting/val  
Outputs pen-lift accuracy (%) and average MDN NLL per batch  

---  

📥 **DOWNLOAD LINKS**  
• DeepWriting Dataset: https://ait.ethz.ch/deepwriting  
• Pretrained Model (best.pth): https://your-domain.com/models/best.pth  

---  

🧑‍💻 **CODE EXPLANATION**  
1. **data.py**  
   – DeepWritingDataset: loads, normalizes & tokenizes strokes + text  
   – collate_fn: pads strokes & text for batched training  

2. **model.py**  
   – MDN-LSTM: TextEncoder, Attention, HandwritingDecoder  
   – SketchRNN: VAE encoder, reparameterization, MDN decoder  

3. **loss.py**  
   – mdn_loss: bivariate-Gaussian MDN + pen-lift BCE  
   – sketchrnn_loss: MDN + categorical CE + KL divergence  

4. **train.py**  
   – CLI with argparse, DataLoaders, tqdm progress bars, validation & checkpointing  

5. **generate.py**  
   – generate_handwriting(): autoregressive sampling of Δx, Δy, pen  
   – CLI prompt to convert text → handwriting  

All code is modular and well-commented—feel free to tune hyperparameters, integrate style embeddings, or experiment with transformer-based variants! ✍️🎉
