# Deep Learning-Based Sign Language Recognition from Videos and Cross-Lingual Translation to Indian Vernaculars
# Sign Language Video Recognition & Cross-Lingual Translation

Deep learning pipeline that classifies sign-language video clips into English word labels and translates them into **Hindi**, **Telugu**, and **Bengali**.

Built as part of an Executive M.Tech project at IIT Patna, under the guidance of Dr. Chandranath Adak.

> 📄 Paper: *Deep Learning-Based Sign Language Recognition from Videos and Cross-Lingual Translation to Indian Vernaculars* — [arXiv link coming soon]

---

## Overview

The system works in two stages:

1. **Recognition** — A fine-tuned [VideoMAE](https://arxiv.org/abs/2203.12602) (`MCG-NJU/videomae-base`) video transformer classifies a 16-frame sign-language video clip into one of 13 English word labels.
2. **Translation** — The predicted English label is translated into Hindi, Telugu, and Bengali using Meta AI's [NLLB-200](https://arxiv.org/abs/2207.04672) (`facebook/nllb-200-distilled-600M`) multilingual translation model.

![Pipeline](arxiv_paper/figures/pipeline.png)

A Streamlit demo app lets a user upload a video and see the predicted label plus all three translations in real time.

---

## Results

Fine-tuned on a 13-class subset (~2GB) of the [AI4Bharat Indian Sign Language dataset](https://openhands.ai4bharat.org/en/latest/instructions/datasets.html) from IIT Madras:

| Metric | Value |
|---|---|
| Training accuracy | 99.4% |
| Validation accuracy | 77.5% (31/40) |
| Classes | 13 |
| Epochs | 15 |

![Confusion Matrix](arxiv_paper/figures/confusion_matrix.png)

Most confusion occurs between visually similar signs (e.g., *hat* / *dress* / *shirt*) and a semantically related cluster (*ugly* / *deaf* / *blind*). See the [paper](./arxiv_paper/main.pdf) for full error analysis.

---
## Setup

```bash
pip install torch torchvision transformers decord scikit-learn
pip install streamlit opencv-python pillow sentencepiece pyngrok
```

Set your Hugging Face token as an environment variable rather than hardcoding it:

```bash
export HF_TOKEN="your_token_here"
export NGROK_AUTH_TOKEN="your_token_here"
```
or add these keys in .env file
---

## Dataset

This project uses a 13-class subset of the AI4Bharat ISL video corpus:

- 8 adjectives: `loud`, `quiet`, `happy`, `sad`, `beautiful`, `ugly`, `deaf`, `blind`
- 5 clothing nouns: `hat`, `dress`, `suit`, `skirt`, `shirt`

Full dataset: [AI4Bharat OpenHands](https://openhands.ai4bharat.org/en/latest/instructions/datasets.html)

Other ISL sources explored (not used for training):
- [Mendeley ISLR dataset](https://data.mendeley.com/datasets/kcmpdxky7p/1)
- [Zenodo ISL corpus](https://zenodo.org/records/4010759)

---

## Training

```python
# Key hyperparameters
Train-Test Split: 80:20
Optimizer: AdamW (lr=1e-5)
Loss: CrossEntropyLoss
Epochs: 15
Batch size: 2
```

See `notebooks/sign_language_classification.ipynb` for the full training loop.

---

## Running the Demo

```bash
streamlit run app.py
```

Upload a `.mp4` or `.mov` clip of a sign from the trained label set. The app returns the predicted English label with a confidence score, followed by Hindi, Telugu, and Bengali translations.

---

## Limitations

- Small, imbalanced label set (13 classes, as few as 1–2 validation clips for some classes)
- Isolated-word classification only — no continuous sentence generation from unsegmented video
- Sensitive to lighting, camera angle, and signer style variation
- Single-word translation can be ambiguous without sentence context
- Offline/batch inference only — not real-time

See the paper's Limitations section for details.

---

## Future Work

- Scale to the full AI4Bharat vocabulary
- Move from isolated-word classification to continuous sign-to-sentence translation
- Extend to more Indian languages via NLLB-200
- Optimize for edge/mobile deployment

---

## Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) and IIT Madras for the ISL dataset
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) open-source teams
- Department of Computer Science and Engineering, IIT Patna

## Citation

```bibtex
@misc{nandipalli2025signlanguage,
  title={Deep Learning-Based Sign Language Recognition from Videos and Cross-Lingual Translation to Indian Vernaculars},
  author={Nandipalli, Ramesh and Adak, Chandranath},
  year={2025},
  note={IIT Patna Executive M.Tech Thesis}
}
```

## License

[Add a license, e.g., MIT — see https://choosealicense.com]
