# Machine Learning for Marketing and Spotify Campaign Analysis

## Recruiter TL;DR
- **Problem:** Marketing and media platforms must make targeting decisions using noisy, high-dimensional data with limited labels.
- **Solution:** Applied supervised and unsupervised machine learning to segment users/content and predict campaign responsiveness.
- **Data:** Spotify 2023 popular songs dataset and a real-world marketing campaign dataset.
- **Models:** K-Means, Gaussian Mixture Models, PCA, ICA, Random Projection, SVM, KNN, and Neural Networks.
- **Key Result:** Segmentation and feature engineering improved rare-response detection in marketing and revealed interpretable Spotify content clusters.
- **Skills Demonstrated:** Marketing analytics, customer segmentation, model selection, hyperparameter tuning, and applied ML reasoning.

---

## Project Overview
This project investigates how **machine learning models must be adapted to different data characteristics** rather than applied uniformly. Using two real-world datasets, I evaluated how supervised and unsupervised methods perform under varying conditions such as class imbalance, feature correlation, dataset size, and noise.

The goal was not just accuracy, but **actionable insight** for:
- Marketing campaign targeting
- Customer segmentation
- Content-based promotion (Spotify-style use cases)

---

## Datasets

### Spotify Songs Dataset (Content & Campaign Segmentation)
- 2023 popular songs with audio and platform-based features
- Target: song *valence* (positivity), binned into five classes
- ~900 observations
- Challenges:
  - High feature correlation
  - Mixed numeric and categorical data
  - Small dataset size for deep learning
- Business framing:
  - Playlist curation
  - Promotional grouping
  - Mood-based content campaigns

### Marketing Campaign Dataset (Customer Targeting)
- Customer demographics, income, spending behavior, and engagement
- Binary target: responsiveness to marketing campaigns
- ~1,700 observations
- Challenges:
  - Strong class imbalance
  - Correlated spending and income features
  - Mixed ordinal and nominal categorical variables
- Business framing:
  - Campaign optimization
  - Cost-efficient targeting
  - Avoiding wasted outreach

---

## Methods

### Unsupervised Learning
- **Clustering**
  - K-Means
  - Gaussian Mixture Models (Expectation Maximization)
- **Dimensionality Reduction**
  - Principal Component Analysis (PCA)
  - Independent Component Analysis (ICA)
  - Random Projection

Used to:
- Discover latent customer and content segments
- Reduce redundancy and noise
- Improve downstream predictive models

---

### Supervised Learning
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Neural Networks**

Each model was evaluated with:
- Train / validation / test splits
- Cross-validation
- Hyperparameter tuning
- Confusion matrices and learning curves

---

## Preprocessing and Feature Engineering
- One-hot encoding for nominal categorical variables
- Target encoding for ordinal features (e.g., education)
- Feature binning (e.g., birth year into eras, valence into classes)
- Standard scaling and bounded normalization
- Removal of identifier and leakage-prone variables
- Separate scaling strategies per dataset to reflect feature semantics

---

## Key Results and Insights

### Spotify Dataset
- Linear SVM performed best due to clean decision boundaries
- Neural networks struggled due to:
  - Small dataset size
  - High noise
  - Weak relationship between popularity signals and positivity
- Clustering revealed interpretable groupings useful for content campaigns
- **Business Impact:** Insights from clusters allow for targeted playlist curation, personalized promotions, and mood-based marketing strategies, helping streaming platforms increase engagement and listener retention.

### Marketing Dataset
- Neural networks outperformed simpler models with sufficient data
- Clustering features improved specificity and rare-response detection
- KNN was sensitive to dimensionality and noise
- SVM provided strong baseline performance with good generalization
- **Business Impact:** Identifying customers who are more likely to engage with campaigns reduces wasted marketing spend, improves ROI, and enables more personalized customer outreach strategies.


---

## Practical Takeaways
- Model complexity must match dataset characteristics
- Unsupervised learning adds value even without large accuracy gains
- Marketing models benefit most from improved identification of *who not to target*
- Feature engineering and preprocessing often matter more than model choice

---

## Technologies Used
- Python 3.10
- scikit-learn
- TensorFlow
- NumPy
- pandas
- SciPy
- Matplotlib

---

## Repository Structure
