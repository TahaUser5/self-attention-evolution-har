# Self-Attention Evolution on HAR Data

Implements and visualizes four progressive levels of self-attention transformation
on multimodal sensor data from the CogAge dataset (Subject: Adeel, Activity: Walking).

## Attention Levels

| Level | Formula |
|-------|---------|
| 1 | Softmax(X · Xᵀ) · X |
| 2 | Softmax((XU)(XU)ᵀ) · XU |
| 3 | Softmax(QKᵀ / √d) · V |
| 4 | Positional Encoding + Level 3 |

## Setup

```bash
pip install numpy pandas matplotlib seaborn
```

## Usage

Place `sensory_data.zip` in your Google Drive at `MyDrive/sensory_data.zip`, then run in Colab:

```python
python attention_har_cogAge.py
```

## Output

Generates a 2×2 heatmap grid showing how attention patterns evolve across the four
levels — from local temporal attention in Level 1 to position-aware global attention
in Level 4.

## Dataset

CogAge / HAR-PR Dataset v2 — multimodal activity recognition using smartphone,
smartwatch, and smart glasses sensor streams (27 axes across 9 sensors).
