# DFCatcher: A Deep CNN Model to Identify Deepfake Face Images

A state-of-the-art deep convolutional neural network for detecting deepfake face images with **98.77% accuracy**.

## Abstract

In recent years, advancement in the realm of machine learning has introduced a feature known as Deepfake pictures, which allows users to substitute a genuine face with a fake one that seems real. As a result, distinguishing between authentic and fraudulent pictures has become difficult. There have been several cases in recent years where Deepfake pictures have been used to defame famous leaders and even regular people. Furthermore, cases have been documented in which Deepfake yet realistic pictures were used to promote political discontent, blackmail, spread fake news, and even carry out false terrorism attacks. The objective of our model is to differentiate between real and Deepfake images so that the above mentioned situations can be avoided. This project represents a deep CNN model with 13000 images divided in two segments that are: Training and Testing. The dataset was prepared using necessary image augmentation techniques. A total of 2 categories are considered (real image category and fake image category). Our suggested model was successful in achieving 98.77% accuracy. The model shows promising results in the case of detecting real and DeepFake images than all the other models used before.

## About

DFCatcher is an 18-layer deep CNN model designed to distinguish between authentic and deepfake face images. As deepfake technology becomes increasingly sophisticated and poses threats to security, privacy, and information integrity, our model provides a robust solution for automated deepfake detection.

### Key Features
- **High Accuracy**: Achieves 98.77% testing accuracy
- **Robust Architecture**: 18-layer CNN with batch normalization and dropout layers
- **Comprehensive Evaluation**: Tested against multiple pre-trained models
- **Real-world Application**: Addresses critical security concerns in digital media

## Performance

| Metric | Score |
|--------|-------|
| Testing Accuracy | 98.77% |
| Training Accuracy | 99.82% |
| Precision | 99.5% |
| Recall | 99.5% |
| F1-Score | 99.5% |

### Comparison with Pre-trained Models

| Model | Testing Accuracy |
|-------|------------------|
| **DFCatcher (Ours)** | **98.77%** |
| InceptionV3 | 97.60% |
| DenseNet121 | 95.50% |
| VGG16 | 89.73% |
| VGG19 | 77.87% |
| ResNet50 | 57.27% |

## Architecture

The DFCatcher model consists of:
- **6 Convolutional layers** with increasing filter sizes (32 → 64 → 128 → 256 → 128 → 64)
- **6 MaxPooling layers** for spatial dimension reduction
- **8 Batch Normalization layers** for training stability
- **3 Dense layers** with dropout for classification
- **Sigmoid activation** for binary classification

Total parameters: 843,521 (841,537 trainable)

## Dataset

The model demonstrates consistent performance across training epochs:

- **Training/Validation Accuracy**: Converges to >98% with minimal overfitting
- **Loss Function**: Binary crossentropy with L2 regularization
- **Optimizer**: Adam with learning rate scheduling

## Technical Details

### Data Augmentation Techniques
- Rescaling, rotation, zoom
- Width/height shifting
- Brightness adjustment
- Horizontal/vertical flipping

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.1 (initial) → 0.000001 (minimum)
- **Batch Size**: 32
- **Epochs**: 20
- **Loss Function**: Binary crossentropy
- **Regularization**: L2 (λ = 0.000001)
- **Callbacks**: ReduceLROnPlateau

## Publication

This work has been published at **IEEE TENCON 2021**:

**"DFCatcher: A Deep CNN Model to Identify Deepfake Face Images"**  
*Arpita Dhar, Likhan Biswas, Prima Acharjee, Shemonti Ahmed, Abida Sultana, Dewan Ziaul Karim, Mohammad Zavid Parvez*

**[Download Paper](https://ieeexplore.ieee.org/document/9707314)** | **[IEEE Xplore](https://doi.org/10.1109/TENCON54134.2021.9707314)**

### Citation
```bibtex
@inproceedings{dhar2021dfcatcher,
  title={DFCatcher: A Deep CNN Model to Identify Deepfake Face Images},
  author={Dhar, Arpita and Biswas, Likhan and Acharjee, Prima and Ahmed, Shemonti and Sultana, Abida and Karim, Dewan Ziaul and Parvez, Mohammad Zavid},
  booktitle={2021 IEEE Region 10 Conference (TENCON)},
  pages={545--550},
  year={2021},
  organization={IEEE},
  doi={10.1109/TENCON54134.2021.9707314}
}
```

## Future Work

- Extension to video deepfake detection
- Mobile application development (Android/iOS)
- Real-time detection capabilities
- Integration with social media platforms
- Multi-language model support

## Authors

- **Arpita Dhar** - *Lead Researcher* - BRAC University
- **Likhan Biswas** - *Co-researcher* - BRAC University  
- **Prima Acharjee** - *Co-researcher* - BRAC University
- **Shemonti Ahmed** - *Co-researcher* - BRAC University
- **Abida Sultana** - *Co-researcher* - BRAC University
- **Awan Ziaul Karim** - *Supervisor* - BRAC University
- **Mohammad Zavid Parvez** - *Co-supervisor* - Engineering Institute of Technology

## Acknowledgments

- BRAC University Department of Computer Science and Engineering
- Engineering Institute of Technology, Melbourne
- IEEE TENCON 2021 Conference
- Kaggle community for datasets

---

**If this work helped your research, please consider citing our paper and giving this repository a star!**
