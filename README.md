# MENet A Mitscherlich Function based Ensemble of CNN Models to Classify Lung Cancer using CT Scans

## Abstract
Lung cancer is one of the leading causes of cancer-related deaths worldwide. To reduce the mortality rate, early detection and proper treatment should be ensured. Computer-aided diagnosis methods analyze different modalities of medical images to increase diagnostic precision. In this paper, we propose an ensemble model, called the Mitscherlich function-based Ensemble Network (MENet), which combines the prediction probabilities obtained from three deep learning models, namely Xception, InceptionResNetV2, and MobileNetV2, to improve the accuracy of a lung cancer prediction model. The ensemble approach is based on the Mitscherlich function, which produces a fuzzy rank to combine the outputs of the said base classifiers. The proposed method is trained and tested on the two publicly available lung cancer datasets, namely Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases (IQ-OTH/NCCD) and LIDC-IDRI, both of these are computed tomography (CT) scan datasets. The obtained results in terms of some standard metrics show that the proposed method performs better than state-of-the-art methods. 


## Contribution
1. We propose an ensemble model, called MENet, for lung cancer classification using CT scans.

2. MENet combines confidence scores obtained from three transfer learning-based CNN models, namely Xception, InceptionResNetV2, and MobileNetV2.

3. We also propose a fuzzy ranking system based on the Mitscherlich function to rank and combine the outputs of different base classifiers for forming an ensemble-based prediction model.
   
4. Our proposed method is trained and tested on two publicly available lung CT scan datasets, namely IQ-OTHNCCD and LIDC-IDRI.

5. MENet outperforms the existing results in lung cancer prediction with an accuracy of 99.54% and 95.75% on IQ-OTHNCCD and LIDC-IDRI datasets, respectively.

## Citation
If you're using this article or code in your research or applications, please consider citing using this BibTeX:

```
@article{Majumder2024,
    doi = {10.1371/journal.pone.0298527},
    author = {Majumder, Surya AND Gautam, Nandita AND Basu, Abhishek AND Sau, Arup AND Geem, Zong Woo AND Sarkar, Ram},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {MENet: A Mitscherlich function based ensemble of CNN models to classify lung cancer using CT scans},
    year = {2024},
    month = {03},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0298527},
    pages = {1-29},
    number = {3},
}
```
