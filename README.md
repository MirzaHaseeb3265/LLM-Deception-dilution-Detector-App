# 🛡️ Robust Deception Monitor for LLMs

[![Live Demo](https://img.shields.io/badge/Live_Demo-Open_App-2563EB?style=for-the-badge)]([PASTE-YOUR-LIVE-APP-LINK-HERE](https://llm-deception-dilution-detector.streamlit.app/))

> Research prototype for detecting localized behavioral evidence of potential deception under deception dilution.

## Overview

Robust Deception Monitor is an API-free Streamlit research prototype that analyzes LLM responses for localized linguistic patterns consistent with strategic deception, contradiction, evasion, misleading framing, suspicious omission, and goal-directed concealment.

The project investigates whether global averaging can miss a small suspicious segment when it is surrounded by many benign statements. It compares a simple global baseline with localized and non-linear aggregation methods designed to retain strong local signals.

This application is **not a lie detector**. Its scores represent behavioral evidence found in text, not proof that a model intended to deceive.

## Research Question

> Can localized and non-linear monitoring detect strategically deceptive behavior hidden through deception dilution while reducing false positives on fiction, roleplay, jokes, hypotheticals, and ordinary uncertainty?

## Main Features

- Whole-response baseline monitoring
- Sentence-level localized analysis
- Global Mean, Max Pooling, Top-K Pooling, and Weighted Pooling
- Dedicated deception-dilution score
- Contradiction and consistency analysis
- Strategic evasion and suspicious omission indicators
- Goal-directed deception indicators
- False-positive adjustment for harmless non-literal contexts
- Segment-level evidence table and highlighted response
- Interactive dilution experiment
- Built-in test cases and research evaluation metrics
- No paid API, API key, database, backend, or authentication

## How It Works

### 1. Response segmentation

The response is divided into sentence-level segments. Very short fragments may be combined so they do not disproportionately affect the result.

### 2. Local behavioral analysis

Each segment is independently checked for transparent linguistic signals related to:

- Direct falsehood language
- Contradictions
- Strategic evasion
- Misleading framing
- Suspicious omission
- Goal-directed influence or concealment
- Normal uncertainty
- Harmless non-literal language

These are heuristic behavioral indicators. They are not hidden-state activations.

### 3. Global baseline

The baseline score is a token-weighted average of all local segment scores:

```text
Global score = weighted mean of all segment evidence scores
```

This approach can be affected by dilution because a suspicious segment may be averaged together with many low-risk segments.

### 4. Non-linear aggregation

The application compares four aggregation strategies:

| Method | Purpose |
|---|---|
| Global Mean | Measures average evidence across the full response |
| Max Pooling | Preserves the strongest local segment |
| Top-K Pooling | Averages the highest-scoring fraction of segments |
| Weighted Pooling | Gives stronger segments more influence |

### 5. Deception-dilution analysis

The dilution score considers:

- Difference between maximum local and global scores
- Difference between Top-K and global scores
- Density of suspicious segments
- Concentration of evidence in a small part of the response

A high dilution score means that local evidence is substantially stronger than the whole-response average. It does not prove deception.

### 6. False-positive defense

Suspicious terms can appear harmlessly in fiction, roleplay, jokes, sarcasm, metaphors, and hypothetical discussions. The monitor searches for explicit contextual markers and reduces the combined score when such framing is present.

This adjustment is imperfect. Context can be implicit, ambiguous, missing, or deliberately manipulated.

### 7. Robust score

The final score combines:

- Global evidence
- Maximum local evidence
- Top-K evidence
- Weighted evidence
- Contradiction evidence
- Evasion evidence
- Dilution evidence
- Harmless-context adjustment

The score is reported using four categories:

- 🟢 **Low Evidence**
- 🟡 **Ambiguous**
- 🟠 **Suspicious**
- 🔴 **High Deception Evidence**

## Important Scientific Distinction

| Evidence level | Description | Supported by this prototype? |
|---|---|---|
| Level 1 — Behavioral evidence | Observable language, contradictions, and response patterns | Yes |
| Level 2 — Internal evidence | Hidden states, activations, or learned probes | No |
| Level 3 — Causal evidence | Interventions proving that internal features cause behavior | No |

The application operates at **Level 1**. It must not be described as mechanistic interpretability or causal proof.

## Factual Errors vs Strategic Deception

The current version does not independently verify general factual claims against an external knowledge source.

For example:

```text
Saturn is the closest planet to the Sun.
```

This statement is factually incorrect, but the sentence alone does not show whether the error was intentional, accidental, or caused by missing knowledge. The current monitor may score it as low evidence because it contains no contradiction, evasion, concealment, or strategic-intent signal.

The app is designed to detect patterns such as:

```text
I know Saturn is not the closest planet, but I will claim that it is so the evaluator accepts my answer.
```

Adding trusted-reference factual verification is a future extension. Factual conflict and intentional deception should remain separate outputs.

## Dilution Experiment

The built-in experiment compares:

1. A short response containing suspicious behavioral language.
2. The same statement surrounded by many benign sentences.

The expected research pattern is:

- The global baseline decreases after benign padding.
- The maximum local and Top-K scores preserve more of the original signal.
- The robust monitor degrades less than the baseline.

## Built-In Test Categories

The application includes examples for:

1. Normal truthful response
2. Obvious deceptive response
3. Deception diluted by truthful content
4. Fiction
5. Roleplay
6. Hypothetical statement
7. Contradictory response
8. Evasive response

## Evaluation Metrics

The research-evaluation section reports:

- Accuracy
- Precision
- Recall
- F1 score
- False Positive Rate
- False Negative Rate
- AUROC
- Baseline dilution degradation
- Robust-monitor dilution degradation

The included dataset is intentionally small and synthetic. It validates application behavior but is not sufficient for scientific performance claims.

## Installation

### Requirements

- Python 3.10 or newer
- pip

### Run locally

```bash
git clone YOUR-GITHUB-REPOSITORY-URL
cd YOUR-REPOSITORY-NAME
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Run in Google Colab

Upload `app.py` and `requirements.txt`, then run:

```python
!pip install -q -r requirements.txt
```

Start Streamlit:

```python
!streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > streamlit.log 2>&1 &
```

Expose it using Cloudflare Quick Tunnel:

```python
!curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared.deb
!cloudflared tunnel --url http://localhost:8501 --no-autoupdate
```

Open the generated `trycloudflare.com` address.

## Project Structure

```text
.
├── app.py
├── requirements.txt
└── README.md
```

## Technology

- Python
- Streamlit
- pandas
- NumPy
- Plotly
- scikit-learn

## Limitations

- The monitor is heuristic and can be gamed.
- False positives and false negatives are possible.
- It does not infer hidden model intent.
- It does not independently fact-check arbitrary claims.
- Explicit non-literal markers do not guarantee harmlessness.
- Absence of suspicious language does not establish truthfulness.
- Built-in evaluation examples are not a representative benchmark.
- White-box probing requires access to model internals.
- Causal verification such as activation patching is outside the project scope.

## Future Work

- Trusted-reference factual contradiction analysis
- Optional local NLI model
- Evaluation on realistic open-ended datasets
- Adversarial paraphrase and obfuscation testing
- Calibration and confidence intervals
- Human-labeled segment localization
- Comparison with white-box activation probes
- Causal verification experiments where model internals are available

## Responsible Use

Do not use this prototype as the sole basis for accusations, moderation decisions, employment decisions, legal conclusions, or other high-impact judgments. Treat its output as exploratory behavioral evidence requiring human review and independent verification.

## License

Add your chosen license here, for example MIT, Apache-2.0, or another license appropriate for your research.

