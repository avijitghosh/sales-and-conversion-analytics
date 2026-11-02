<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Segoe+UI&weight=600&size=28&duration=4000&pause=1000&color=007ACC&center=true&vCenter=true&width=700&lines=Sales+Prediction+%26+Conversion+Analytics;Predictive+Modeling+%26+Classification+Pipeline" alt="Typing SVG" />
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Language-Script-blue?style=flat-square" alt="Language"></a>
  <a href="#"><img src="https://img.shields.io/badge/Focus-Predictive_Analytics-3776AB?style=flat-square" alt="Focus"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Completed-success?style=flat-square" alt="Status"></a>
</p>

---

## 📈 Executive Summary
This analytics project tackles two foundational tasks: **Sales Prediction** (Continuous Outcome Modeling) and **Customer Conversion** (Binary Classification) using dynamic marketing datasets[cite: 4, 8]. By implementing a 70/30 train-validation split (reproducible seed structure), multiple linear regression models and logistic regression architectures were constructed, evaluated, and optimized based on validation set performance metrics[cite: 5, 7, 9, 13].

---

## 🔄 Analytical Pipeline

```mermaid
graph TD;
    A[Raw Data Audit: Sales & Conversion CSVs] --> B[Reproducible Data Split: 70% Train / 30% Validation];
    B --> C[Prediction Track: Multiple Linear Regression];
    B --> D[Classification Track: Logistic Regression];
    C --> E[Feature Selection & Outlier Mitigation];
    D --> F[AIC Optimization & Probability Modeling];
    E --> G[Model Evaluation: RMSE, ME, MAD, MPE];
    F --> H[Classifier Validation: ROC-AUC & Confusion Matrix];
