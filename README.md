# Xvertice Pipeline

An explainable image forensics system for detecting manipulated, AI-generated, and AI-edited images using a combination of deep learning and digital forensic analysis.

## Quick Start

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/pri2304/xvertice-pipeline.git
cd xvertice-pipeline

pip install -r requirements.txt
```

---

### Running the API Server

Start the FastAPI server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Or during development:

```bash
uvicorn server:app --reload
```

Once running, the API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```
To analyse an image through the complete pipeline:

1. Start the API server.
2. Submit an image through the API endpoint.
3. Receive:

   * Prediction score
   * Forensic feature outputs
   * SHAP explanations
   * Structured JSON report
   * Visual results in [Results](Results/)
   
4. You can pass the JSON into an LLM and ask it to interpret the JSON for a human understandable explanation.

Example output of JSON:
[sample_response.json](examples/sample_response.json).

---

### Running Individual Forensic Tests

Each forensic module can be executed independently for experimentation and analysis. Just change the input filepath in the file itself
```python
    input_path = "testimage.jpg" # <- Change to image you want to test
```

#### DQT-Aware ELA

```bash
python forensic_tests.DQT_Aware_ELA.py
```

#### Metadata Analysis

```bash
python forensic_tests.metadata_analysis_final.py
```

#### Noise Analysis

```bash
python forensic_tests.noise_analysis_test.py
```

#### NLF Analysis

```bash
python forensic_tests.NLF.py
```

#### DCT Frequency Analysis

```bash
python forensic_tests.DCT.py
```

#### GAN Frequency Analysis

```bash
python forensic_tests.GAN_frequency.py
```

#### JPEG Ghost Analysis

```bash
python forensic_tests.jpeg_ghost_analysis.py
```
#### CNN Extraction

```bash
python feature_extraction.cnn_extraction.py
```
---

## Overview

Traditional image detectors often provide only a confidence score without explaining why a prediction was made. Xvertice was built to address this problem by combining multiple forensic signals with machine learning to produce both a prediction and supporting evidence.

The system extracts features from a CNN alongside a suite of automated forensic tests, then aggregates these signals using an XGBoost ensemble model. A SHAP-based explainability layer is used to identify which signals contributed most strongly to the final prediction.

---

## Key Features

* Detection of manipulated, AI-generated, and AI-edited JPEG images
* CNN-based feature extraction
* Ensemble prediction using XGBoost
* SHAP-powered explainability
* Multiple automated forensic tests
* Structured JSON outputs for downstream LLM analysis
* Visual forensic outputs for human investigation

---

## Forensic Analysis Modules

The pipeline combines multiple complementary forensic approaches.

### Compression Analysis

* DQT-aware Error Level Analysis (ELA)
* JPEG Ghost Analysis
* JPEG Block Artifact Grid (BAG) metrics

### Metadata Analysis

* EXIF consistency checks
* Software trace detection
* Camera metadata extraction
* Timestamp validation
* Metadata stripping detection
* Binary signature scanning

### Noise & Residual Analysis

* Laplacian residual analysis
* Median residual analysis
* Local variance analysis
* Sensor noise consistency analysis

### Frequency Domain Analysis

* DCT-based frequency statistics
* FFT/GAN frequency analysis
* High-frequency energy measurements

### Noise Level Function (NLF)

* Camera noise profile consistency estimation
* Detection of anomalous local regions

---

## System Architecture

Input Image
↓
CNN Feature Extraction
↓
Forensic Feature Extraction
↓
Feature Aggregation
↓
XGBoost Ensemble
↓
SHAP Explainability
↓
Prediction + Evidence Output

---

## Model Experimenting

A significant portion of the project focused on understanding which signals contributed most effectively to manipulation detection.

Experiments included:

* Individual feature evaluation
* Single-feature model training
* Feature pair testing
* Feature group ablation studies
* CNN-only evaluation
* Forensics-only evaluation

These experiments were used to identify the most informative features and guide the design of the final ensemble model.

One of the key findings was that no single forensic signal consistently performed well across all manipulation types. Combining CNN features with forensic features produced significantly more robust results than either approach alone.

---

## Explainability

Rather than treating the model as a black box, Xvertice integrates SHAP to provide feature-level explanations.

This allows investigators to understand:

* Which forensic signals influenced the prediction
* Whether metadata anomalies contributed
* Whether frequency-domain artifacts were detected
* Whether noise inconsistencies were present
* How strongly each signal affected the final score

---

## Technologies Used

* Python
* PyTorch
* XGBoost
* OpenCV
* NumPy
* Pandas
* SHAP
* FastAPI
* Scikit-learn
* Efficientnet B4 (As CNN base model)

---

## Dataset

Training datasets and generated feature tables are not included in this repository due to size limitations.

The project was trained and evaluated using large-scale collections of authentic, manipulated, and AI-generated images. Dataset preparation, feature engineering methodology, and training procedures are documented separately and can be discussed upon request.

---

## Demonstration

A detailed demonstration video of the system is available through the project's public posts and showcases:

* End-to-end image analysis
* Forensic visualizations
* Explainability outputs
* Model predictions
* Reporting workflow

[Link to Demo](https://www.linkedin.com/feed/update/urn:li:activity:7467541821951307776)

---

## Repository Structure

```text
xvertice-pipeline/
├── Models/
├── Results/
├── Training/
├── api/
├── examples/
├── feature_extraction/
├── forensic_tests/
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```
---

## Limitations

* Optimized primarily for JPEG images
* Social media recompression can affect forensic signals
* No single forensic test is reliable in isolation
* Performance may vary across unseen manipulation techniques or newer generative models

---

## Future Work

* CLI for singular or batch and testing entire pipeline locally, along with testing new models
* Improved social-media robustness
* Additional forensic modalities
* Expanded benchmark evaluation
* Improved inference tooling
* Enhanced reporting and visualization pipelines
