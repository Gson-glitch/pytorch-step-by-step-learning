# Deep Learning with PyTorch — Learning Repository

This repository contains my personal notes, experiments, and practice code as I work through **[_Deep Learning with PyTorch: Step-by-Step, A Beginner's Guide_ by Daniel Voigt Godoy (v1.2)](https://github.com/dvgodoy/PyTorchStepByStep)**.

The goal of this repo is to track my learning progress and document hands-on exercises, experiments, and supporting scripts.

---

## 📚 Project Structure

```
.
├── chapters/                 # Jupyter notebooks for each chapter of the book
│   ├── chapter_0/
│   ├── chapter_1/
│   ├── chapter_2/
│   ├── chapter_3/
│   └── chapter_4/
├── data_generation/          # Scripts to generate synthetic datasets
├── data_preparation/         # Data preparation utilities
├── general_knowledge/        # Additional notebooks exploring ML/DS concepts
├── model_configuration/      # Model architecture definitions (e.g., v0, v1, v2...)
├── model_training/           # Training loops, experiments, and versions
├── stepbystep/               # Utility modules following the book's structure
├── runs/                     # TensorBoard logs and experiment outputs
├── modal_jupyter_server.py   # Helper script for running Jupyter remotely
├── model_checkpoint.pth      # Saved model checkpoint
├── pyproject.toml            # Project configuration
└── uv.lock                   # Dependency lockfile

```

---

## 🚀 Goals
- Follow the book chapter by chapter
- Implement all examples in notebooks and scripts
- Explore variants and run experiments to deepen understanding
- Build utilities for data generation, preprocessing, and model training
- Track experiments using TensorBoard

---

## 🛠️ Environment Setup
This project uses **Python** and is managed via **uv** (or pip if preferred).

### Install dependencies
```
uv sync
```
Or, using pip:
```
pip install -r requirements.txt
```

### Launch Jupyter Notebook
```
uv run jupyter notebook
```

---

## 📈 Experiment Tracking
Training runs and metrics are logged under the `runs/` directory.
To visualize training progress with TensorBoard:
```
tensorboard --logdir runs/
```

---

## 🔍 Additional Learning Material
The `general_knowledge/` folder includes notebooks on:
- Gradient descent
- Correlation and quantiles
- Decision trees from scratch
- Data analysis basics
- Encoding, regex, dependency injection, meshgrid, groupby, and more

These are side explorations to strengthen foundational understanding.

---

## 🤝 Acknowledgments
This learning repository is based on the book:
[_Deep Learning with PyTorch: Step-by-Step, A Beginner's Guide_ by Daniel Voigt Godoy (v1.2)](https://github.com/dvgodoy/PyTorchStepByStep).

---

## 📌 Notes
This repo is a work in progress. I will continue adding notebooks, experiments, and learning notes as I progress through the book.

