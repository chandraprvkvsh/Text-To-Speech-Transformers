# Text to Speech Transformers

Implemented Tacotron 2 and Transformer TTS models from scratch for high-quality speech synthesis using the LJ Speech dataset.

---

## 1. Audio Signal Analysis

### Waveform Visualization
- Displays amplitude variations over time to understand the dynamic range, periodicity, and transient events in the speech signal.

### Spectrogram Analysis
- Visualizes frequency content over time, revealing harmonic structures, frequency distribution, and spectral energy dynamics.

### Power Spectrum Analysis
- Represents spectral energy distribution, providing insights into the tonal richness and spectral characteristics.

### Autocorrelation Analysis
- Measures self-similarity in the signal to identify fundamental frequency and periodic structures crucial for speech synthesis.

### Envelope Extraction
- Captures amplitude dynamics for speech segmentation and emotion recognition by extracting the signal's envelope.

### Audio Feature Extraction
Using the `librosa` library, the following features are extracted:
- **Chroma Feature (chroma_stft)**: Pitch class distribution.
- **MFCCs**: Spectral characteristics for speech analysis.
- **Spectral Centroid**: Center of mass of the power spectrum.
- **Spectral Bandwidth**: Frequency spread.
- **Spectral Contrast**: Peaks and valleys differentiation in frequency bands.
- **Spectral Rolloff**: Frequency below which most spectral energy is concentrated.
- **Zero Crossing Rate**: Sign change rate, related to noisiness.

### Pitch Estimation
- Utilizes `librosa.piptrack()` to extract pitch variations over time, visualizing them in MIDI notation.

### Mel-Spectrogram Generation
- Transforms the spectrogram into a perceptually relevant representation, crucial for speech recognition and synthesis tasks.

---

## 2. Deep Learning-Based Speech Synthesis

### Tacotron 2 Implementation (Built from Scratch)
Tacotron 2 is implemented from scratch, following the original paper, to generate high-quality mel-spectrograms from text sequences. Key components include:

#### Encoder PreNet Layer
- **ReLU Activation & Dropout**: Enhances generalization.
- **Dense Layer Transformation**: Improves feature representation before the encoder.

#### CBHG Layer Implementation
- **Convolutional Filters**: Extract local features with varying kernel sizes.
- **Max Pooling**: Downsamples feature maps.
- **Highway Networks**: Enhances non-linear transformations.
- **Bidirectional LSTM**: Captures long-term dependencies in speech signals.

---

### Transformer TTS Implementation (Built from Scratch)
A Transformer-based TTS model is implemented from scratch, inspired by the latest research papers, achieving high-quality, natural-sounding speech synthesis with the following features:

- **Self-Attention Mechanisms**: Captures global dependencies in speech sequences.
- **Positional Encoding**: Retains temporal order of input sequences.
- **High-Quality Speech Outputs**: Produces natural and intelligible speech, leveraging the power of transformers.

---

## 3. Key Contributions

- Comprehensive analysis of the LJ Speech dataset using advanced audio processing techniques.
- Implementation of **Tacotron 2** and **Transformer TTS** from scratch, ensuring a deeper understanding of sequence-to-sequence modeling for speech synthesis.
- Utilization of state-of-the-art deep learning architectures, including CBHG layers, self-attention mechanisms, and positional encodings.

---

## 4. Presentation and Notebook

For a comprehensive understanding of the project, please refer to the presentation included in the repository. Additionally, detailed comments have been added to the notebook, offering insights into the code and methodology.

---

## 5. Note on Training Constraints

Due to constraints such as limited time and computational resources, the models were trained under constrained conditions, which may impact performance. We appreciate your understanding and plan to update the repository or create a new one as progress is made. Training transformer models from scratch is resource-intensive, and we apologize for any inconvenience caused by this limitation.

---


## 6. Future Work
- Training on larger datasets for enhanced generalization.
- Fine-tuning Transformer TTS models for expressive speech synthesis.
- Exploring advanced speech enhancement techniques for noise reduction.

---

## 7. Acknowledgments
- Implementations are based on the original research papers for Tacotron 2 and Transformer TTS, built from scratch for educational and research purposes.
- Special thanks to the developers of the `librosa` and `PyTorch` libraries for facilitating audio processing and deep learning model development.

---

## 8. References
- Tacotron 2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions
- Transformer TTS: Non-Autoregressive Parallel Text-to-Speech
