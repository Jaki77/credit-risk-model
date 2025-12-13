# Credit Risk Probability Model for Alternative Data

## Project Overview

This project implements an end-to-end credit risk probability model for **Bati Bank**'s new "buy-now-pay-later" service in partnership with an e-commerce platform. The model uses alternative data (transaction history) to predict creditworthiness, creating a proxy target variable through RFM (Recency, Frequency, Monetary) analysis and K-Means clustering.

## Business Context

Bati Bank, a leading financial service provider with over 10 years of experience, is launching a buy-now-pay-later service with an e-commerce partner. Traditional credit scoring approaches require historical loan repayment data, which is unavailable in this scenario. Instead, we leverage customer transaction behavior from the e-commerce platform to build a predictive credit risk model.

---

# Credit Scoring Business Understanding

## 1. Basel II Accord's Influence on Model Interpretability

The **Basel II Capital Accord** fundamentally shifted banking regulation from a one-size-fits-all approach to a **risk-sensitive framework** that emphasizes **internal risk measurement**. This has several critical implications for our model:

### Key Influences:

#### Pillar 1 - Minimum Capital Requirements
Basel II requires banks to hold capital reserves proportional to their risk exposure. An inaccurate credit risk model could lead to:
- **Underestimation of risk** → Inadequate capital buffers → Potential bank insolvency during economic downturns
- **Overestimation of risk** → Excessive capital allocation → Reduced profitability and lending capacity

#### Pillar 2 - Supervisory Review
Regulators require banks to have robust **internal capital adequacy assessment processes** (ICAAP). Our model must be:
- **Transparent** and explainable to both internal stakeholders and regulators
- **Well-documented** with clear methodology, assumptions, and limitations
- **Validated** through back-testing and stress-testing procedures

#### Pillar 3 - Market Discipline
Requires public disclosure of risk management practices. This creates reputational risk if models are poorly designed or documented.

### Interpretability Imperative
In this regulated context, we need models where:
- **Decisions can be explained** to customers who are denied credit
- **Regulators can audit** the logic and fairness of credit decisions
- **Business stakeholders understand** risk drivers to make informed policy decisions

---

## 2. Proxy Variable Necessity and Associated Business Risks

### Why a Proxy Variable is Necessary:

1. **Data Gap**: The eCommerce dataset contains transaction history but no direct loan repayment records. Without a proxy, we cannot apply supervised learning techniques.

2. **Behavioral Similarity Principle**: Customers who exhibit poor engagement patterns (low frequency, low spending, irregular transactions) are likely to exhibit similar patterns with loan repayments.

3. **RFM as Risk Indicator**: Recency, Frequency, and Monetary patterns have been empirically shown to correlate with creditworthiness in alternative credit scoring literature.

### Potential Business Risks of Proxy-Based Predictions:

#### A. Model Risk:
- **Proxy Misalignment Risk**: RFM patterns may not perfectly correlate with actual default behavior. An active shopper ≠ a reliable borrower.
- **Concept Drift**: Ecommerce behavior patterns may change over time, becoming less predictive of credit risk.

#### B. Business Impact Risks:

**1. False Positives (Type I Error):**
   - **Rejecting credit-worthy customers** → Lost revenue and customer alienation
   - **Particularly harmful** for a new "buy-now-pay-later" service trying to gain market share

**2. False Negatives (Type II Error):**
   - **Approving high-risk customers** → Loan defaults, financial losses
   - **For a new service**, early defaults could damage reputation and investor confidence

**3. Regulatory Risks:**
   - **Discrimination Risk**: If RFM patterns correlate with protected characteristics (age, location), the model could inadvertently create biased outcomes
   - **Validation Challenges**: Regulators may question the validity of proxy-based models compared to traditional credit scoring

#### C. Strategic Risks:
- **Market Selection Bias**: The model may systematically exclude certain customer segments, limiting market penetration
- **Over-reliance on Digital Footprint**: Customers with limited eCommerce activity (older demographics, rural populations) may be unfairly penalized

---

## 3. Model Selection Trade-offs: Simple vs. Complex Models

### Comparison Table

| **Aspect** | **Simple Model (Logistic Regression with WoE)** | **Complex Model (Gradient Boosting)** |
|------------|-------------------------------------------------|---------------------------------------|
| **Interpretability** | **High**: Linear relationships, clear feature importance via WoE/IV | **Low**: "Black box" nature, complex interactions |
| **Regulatory Compliance** | **Easier**: Decisions can be explained using point-based scoring systems | **Challenging**: Requires additional techniques (SHAP, LIME) for explanation |
| **Performance** | **Lower**: May miss non-linear relationships and complex interactions | **Higher**: Can capture complex patterns and interactions |
| **Implementation** | **Straightforward**: Established methodology, easier to validate | **Complex**: More hyperparameters, harder to debug |
| **Maintenance** | **Easier**: Stable, predictable updates | **Harder**: Sensitive to data drift, retraining complexity |
| **Audit Trail** | **Clear**: Each feature's contribution is quantifiable | **Opaque**: Feature importance available but causal relationships unclear |
| **Business Adoption** | **Higher**: Stakeholders understand and trust the logic | **Lower**: Requires education on model limitations |
| **Model Governance** | **Simpler**: Easier to document, validate, and monitor | **Complex**: Requires sophisticated MLOps infrastructure |

### Strategic Considerations for Bati Bank:

#### Arguments for Logistic Regression with WoE:
1. **Regulatory Safety**: In a heavily regulated sector, explainability often trumps marginal performance gains
2. **Basel II Alignment**: The accord emphasizes understanding and quantifying risk drivers—easier with interpretable models
3. **Scoring Card Tradition**: The finance industry has decades of experience with scorecards, easing adoption
4. **Dispute Resolution**: When customers question decisions, linear models provide clear "reasons for decline"

#### Arguments for Gradient Boosting:
1. **Competitive Advantage**: In a new market, better risk discrimination could provide a significant edge
2. **Alternative Data Strength**: Complex models may better extract signals from non-traditional data sources
3. **Post-Hoc Explanations**: Techniques like SHAP can provide acceptable explanations for regulators
4. **Future-Proofing**: As digital footprints grow, complex patterns become more important

### Recommended Hybrid Approach for Bati Bank:

Given that this is a **new service** with **no historical default data**, I recommend:

**Phase 1: Start Simple**
Begin with Logistic Regression + WoE to establish:
- Regulatory credibility
- Clear business rules
- Baseline performance

**Phase 2: Phase in Complexity**
Once the service matures and we collect actual repayment data:
- Use Gradient Boosting as a "challenger model"
- Compare performance rigorously
- If significantly better, invest in explanation infrastructure

**Phase 3: Model Governance Framework**
Regardless of choice, implement:
- Regular validation and monitoring
- Bias and fairness testing
- Clear documentation of model limitations
- Human oversight for borderline cases

---

## Conclusion

The Basel II framework makes **model transparency non-negotiable** for Bati Bank. While proxy variables enable credit scoring with alternative data, they introduce **significant validation and fairness challenges**. The choice between simple and complex models represents a **strategic balance** between regulatory compliance and competitive performance in Ethiopia's evolving financial landscape.

Given the **newness of the service** and **regulatory scrutiny**, starting with an interpretable model provides a **safer foundation** that can be enhanced with more sophisticated techniques as data and regulatory comfort grow.

---

## Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml           # CI/CD pipeline
├── data/                              # (ignored in git)
│   ├── raw/                           # Raw data
│   └── processed/                     # Processed data
├── notebooks/eda.ipynb                # EDA notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py             # Feature engineering
│   ├── train.py                       # Model training
│   ├── predict.py                     # Inference script
│   └── api/
│       ├── main.py                    # FastAPI app
│       └── pydantic_models.py         # API schemas
├── tests/test_data_processing.py      # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md                          # Business understanding & documentation

```