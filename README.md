# 🚀 **Pico Train** but with **ReLoRA**

Pico Train is a lightweight framework for training language models—from tiny-scale (~1M parameters) to mid-scale (~1B parameters)—with built-in rich checkpointing that captures activations, gradients, and model states, enabling detailed learning dynamics research.

Our **suite of pre-trained models** is already publicly available on our [Hugging Face organization](https://huggingface.co/pico-lm), and a dedicated companion library for advanced analysis—[**pico-analyze**](https://github.com/pico-lm/pico-analyze)—is fully released for deeper checkpoint studies.

> For a **detailed run-through**, check out the **full tutorial** on our website at [picolm.io](https://picolm.io).

---

## **Key Features**

1. **Pico Decoder: LLAMA-style Transformer Architecture**  
   - RMSNorm, RoPE, multi-head self-attention with KV-cache, and SwiGLU activations  
   - Currently supports the **pico-decoder** model, with future expansions planned (pico-diffusion, pico-statespace, etc.)
   - Integrated and configurable ReLoRA integration (see point 5)

2. **Comprehensive Checkpoints**  
   - Saves model states, optimizer states, and training metadata  
   - Enriched with **activation and gradient** snapshots for interpretability  

3. **Focused Scale Range**  
   - Optimized to train models from **1M to 1B parameters**, where learning dynamics research is most viable  

4. **Clean, Pre-tokenized Data**
   - Uses a pre-tokenized, pre-shuffled version of [Dolma](https://allenai.org/dolma) that we make available on [Hugging Face](https://huggingface.co/datasets/pico-lm/pretokenized-dolma)  
   - Facilitates training models using identical data for **consistency** and **comparability**

5. **ReLoRA Integration**
   - Efficient parameter-efficient training with periodic merging
   - Configurable target modules for selective ReLoRA application
   - Trainable scaling factors for adaptation flexibility
   - Compatible with distributed training and optimizers

6. **Research Ready**  
   - Minimal, well-documented code suitable for **forking and tailoring**  
   - Logs essential metrics (e.g. perplexity) throughout training  
   - Works seamlessly with [pico-analyze](https://github.com/pico-lm/pico-analyze) for advanced post-training interpretation

---

## **ReLoRA Features**

The ReLoRA implementation provides several key capabilities:

- **Selective Adaptation**: Target specific layers for ReLoRA training
- **Configurable Reset Frequency**: Control when weights are merged and reinitialized
- **Flexible Architecture**:
  - Low-rank adaptation matrices (A and B)
  - Optional trainable scaling factors

### **ReLoRA Configuration**

Configure ReLoRA through the model config:

```yaml
model:
  relora:
    target_modules: ["swiglu", "attention"]     # Layers to apply ReLoRA
    reset_frequency: 1000                       # Steps between merges
    r: 128                                      # Low-rank adaptation dimension
    lora_alpha: 32                              # LoRA scaling factor
    lora_dropout: 0.1                           # Dropout probability
    keep_original_weights: true                 # Keep base model weights
    trainable_scaling: false                    # Fixed vs learned scaling
```

## **Training Philosophy**

All models in the Pico suite (both pre-trained and user-trained):

- Employ **identical architectures** and **optimizer settings**  
- **Share** the same data order and tokens  
- Automatically log **rich checkpoint data** (including activations, gradients)  
- Facilitate **direct cross-scale comparisons**

This uniformity means you can isolate model size as the primary variable, giving you clearer insights into **how model capacity affects learning**.

---

## 📦 Resources

All our pre-trained models and datasets are publicly available through our [HuggingFace organization](https://huggingface.co/pico-lm):

- Pre-trained models (1M to 1B parameters)
- Pre-tokenized training data derived from the DOLMA corpus
- Training checkpoints with activation and gradient information
- Basic evaluation (perplexity) metrics logged throughout training

## 🌟 Why Pico?

Unlike other model suites, Pico is specifically designed for learning dynamics research:

1. **Focused Scale Range**: Covers the critical 1M-1B parameter range where most learning dynamics research is feasible
2. **Consistent Training**: All models see identical data in identical order, enabling true cross-scale comparisons
3. **Rich Analytics**: Automatic saving of activations and gradients for mechanistic interpretability
4. **Research Ready**: Minimal, well-documented code designed to be forked and modified
5. **Clean Data**: Uses a curated, pre-shuffled version of the DOLMA corpus
6. **Train Your Own**: Simple pipeline for training your own suite of models with custom configurations

## 🔑 Key Features

- **Simple Architecture**: Clean, modular implementation of core transformer components
- **Educational Focus**: Well-documented code with clear references to academic papers
- **Research Ready**: Built-in tools for storing and studying model learning dynamics
- **Efficient Training**: Pre-tokenized dataset and optimized training loop
- **Modern Stack**: Built with PyTorch Lightning, Wandb, and HuggingFace integrations

## 🏗️ Core Components

- **RMSNorm** for stable layer normalization
- **Rotary Positional Embeddings (RoPE)** for position encoding
- **Multi-head attention** with KV-cache support
- **SwiGLU activation** function
- **Residual connections** throughout

## 🚀 Quick Start

1. **Clone Project**

   ```bash
   git clone https://github.com/pico-lm/pico-train
   cd pico-train
   ```

2. **Configure Environment**

   Create a `.env` file at the root with your Hugging Face and Weights & Biases tokens:

   ```bash
   export HF_TOKEN=your_huggingface_token
   export WANDB_API_KEY=your_wandb_key
   ```

3. **Install Dependencies**

   ```bash
   source setup.sh
   ```

   This script checks your environment, installs necessary tools, and sets up a Poetry virtual environment.

4. **Train Your Model Suite**

   - Edit (or create) a config file (e.g., `configs/demo.yaml`) to specify your architecture and training preferences.
   - Then run:

     ```bash
     poetry run train --config_path configs/demo.yaml
     ```

   - This launches training, automatically checkpointing states and saving learning dynamics data.

5. **Explore Checkpoints**
   - By default, checkpoints are stored under `runs/YOUR_RUN_NAME/checkpoints/`.
   - Each checkpoint contains:
     - **Model state** (PyTorch + Hugging Face formats)
     - **Optimizer state**
     - **Gradients and activations** for interpretability
     - **Evaluation logs** (e.g. perplexity) and metrics

---

## **Repository Structure**

- **`src/model/pico_decoder.py`**  
  - Core LLAMA-style decoder implementation (attention, RMSNorm, RoPE, etc.)

- **`src/training/trainer.py`**  
  - Main training loop  
  - Manages distributed and multi-node settings  
  - Collects/logs metrics  
  - Orchestrates checkpoint saving

- **`src/checkpointing`**  
  - Logic for saving model states, gradients, activations  
  - Tools for uploading checkpoints to Hugging Face

- **`src/config`**  
  - Flexible Dataclass-based config system (model and training hyperparameters, checkpointing, logging)

- **`configs/demo.yaml`**  
  - Example config with default values for quick experimentation

---

## **Advanced Analysis with Pico Analyze**

For deeper checkpoint analysis—comparing gradients, tracking representation shifts, measuring sparsity—use our companion repository [**pico-analyze**](https://github.com/pico-lm/pico-analyze). It automatically processes **pico-train** checkpoints and applies advanced metrics like **CKA**, **PWCCA**, **Gini**, **Hoyer**, and more to reveal **how** your models learn over time.

---

## **License**

Pico is open-source under the [Apache License 2.0](LICENSE).

---

## **Citation**

If you use **Pico** in your research, please cite:

```bibtex
@software{pico-relora-2025,
    author = {Diehl Martinez, Richard and Weiss, Yuval},
    title = {Pico ReLoRA: A Lightweight Framework for Studying Language Model Learning Dynamics, with integrated ReLoRA training support},
    year = {2025},
    url = {https://github.com/Yu-val-weiss/relora-pico}
}
```

**Happy Training!** For more information and tutorials, visit our website at [picolm.io](https://picolm.io).
