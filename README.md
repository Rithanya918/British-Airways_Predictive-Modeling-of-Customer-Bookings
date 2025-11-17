
# British Airways Customer Booking Prediction

## Project Overview

A machine learning solution that predicts customer flight booking completion for British Airways. The model identifies **78% of customers who will complete bookings**, enabling proactive marketing strategies and improved customer acquisition.

###  Business Problem

In today's digital landscape, customers research and book flights online before arriving at airports. Traditional reactive sales strategies are no longer effective. Airlines need to:

- Identify potential customers early in their journey
- Target high-conversion customer segments
- Optimize marketing spend with data-driven decisions
- Shift from reactive to proactive customer engagement

https://www.theforage.com/virtual-experience/NjynCWzGSaWXQCxSX/british-airways/data-science-yqoz/predicting-customer-buying-behaviour

### Solution

Built a **Random Forest Classifier** that analyzes customer behavior patterns and flight preferences to predict booking completion with:

- **77.74% Recall** - Catches nearly 8 out of 10 potential bookings
- **76.61% ROC-AUC** - Strong discriminative ability
- **Actionable Insights** - Identifies key factors driving conversions

---

## Dataset

- **Size:** 50,000 customer records
- **Features:** 14 original features (expanded to 934 after engineering)
- **Target:** Binary classification (Booking Complete: Yes/No)
- **Class Distribution:** 85% No Booking, 15% Booking (Imbalanced)

### Original Features

| Feature | Description |
|---------|-------------|
| `num_passengers` | Number of passengers in booking |
| `sales_channel` | Channel used for booking (Internet/Mobile) |
| `trip_type` | Type of trip (RoundTrip/OneWay/CircleTrip) |
| `purchase_lead` | Days between booking and flight |
| `length_of_stay` | Number of nights at destination |
| `flight_hour` | Hour of flight departure |
| `flight_day` | Day of week for flight |
| `route` | Flight route code |
| `booking_origin` | Country where booking was made |
| `wants_extra_baggage` | Customer wants extra baggage (0/1) |
| `wants_preferred_seat` | Customer wants preferred seat (0/1) |
| `wants_in_flight_meals` | Customer wants in-flight meals (0/1) |
| `flight_duration` | Total flight time in hours |
| `booking_complete` | **Target variable** (0/1) |

---

## Feature Engineering

Created **15 new features** to capture hidden patterns:

```python
# Examples of engineered features
df['is_weekend_flight'] = df['flight_day'].apply(lambda x: 1 if x in ['Sat', 'Sun'] else 0)
df['total_extras'] = df['wants_extra_baggage'] + df['wants_preferred_seat'] + df['wants_in_flight_meals']
df['route_popularity'] = df['route'].map(df['route'].value_counts())
df['is_long_haul'] = (df['flight_duration'] > 6).astype(int)
```

| Feature | Type | Business Logic |
|---------|------|----------------|
| `is_weekend_flight` | Binary | Travel behavior patterns |
| `booking_lead_category` | Categorical | Booking urgency indicator |
| `stay_category` | Categorical | Trip purpose indicator |
| `total_extras` | Numerical | Customer value indicator |
| `route_popularity` | Numerical | Popular destination indicator |
| `origin_popularity` | Numerical | Market strength indicator |
| `is_long_haul` | Binary | Flight type classification |
| `is_solo_traveler` | Binary | Travel pattern |
| `is_family_trip` | Binary | Group dynamics |
| `is_high_value_customer` | Binary | Revenue potential |

---

## Model Architecture

### Algorithm: Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,           # 100 decision trees
    class_weight='balanced',    # Handle class imbalance
    max_depth=10,               # Prevent overfitting
    min_samples_split=10,       # Minimum samples to split
    min_samples_leaf=5,         # Minimum samples in leaf
    random_state=42,            # Reproducibility
    n_jobs=-1                   # Parallel processing
)
```

### Why Random Forest?

✅ **Interpretability** - Provides feature importance scores  
✅ **Handles Imbalance** - Built-in class weighting  
✅ **Robust** - Resistant to outliers and noise  
✅ **No Scaling Required** - Scale-invariant algorithm  
✅ **Feature Importance** - Explains which factors drive predictions  

---

## Results

### Model Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 66.70% | Overall correct predictions |
| **Precision** | 27.96% | When predicting booking, correct 28% of time |
| **Recall** | **77.74%** | **Catches 78% of actual bookings** |
| **F1-Score** | 41.12% | Balance of precision and recall |
| **ROC-AUC** | **76.61%** | **Strong discriminative ability** |

### Why Recall > Accuracy?

High recall is prioritized because:
- **Missing bookings = Lost revenue** (high cost)
- **False positives = Extra marketing emails** (low cost)
- **Business value**: Capturing 78% of potential customers is more valuable than 85% accuracy that misses most bookings

### Top Predictive Features
<img width="1226" height="781" alt="Image" src="https://github.com/user-attachments/assets/d87b71a4-578f-4f64-8436-e7902da406df" />

**Key Insight:** Geographic origin is the strongest predictor - certain markets convert significantly better than others!

---

##  Key Visualizations

### Class Distribution
<img width="1095" height="495" alt="Image" src="https://github.com/user-attachments/assets/5b550826-271e-4638-90b7-fb6ce7aa340a" />

- 85% customers don't complete booking
- 15% customers complete booking
- Imbalance ratio: 5.69:1

### Confusion Matrix
<img width="1355" height="494" alt="Image" src="https://github.com/user-attachments/assets/a89878b6-54e3-462c-aadf-41caf228e5c3" />

- True Positives: ~1,163 bookings correctly identified
- Successfully captures 77.7% of revenue opportunities

---

## Business Recommendations

### 1. **Target High-Converting Markets**
- Increase marketing investment in **Australia** and **Malaysia**
- These markets show highest conversion rates
- Expected ROI: 20-30% improvement in conversion

### 2. **Deploy Predictive Model**
- Score all customers with booking probability
- Flag customers with >50% probability
- Enable early intervention campaigns

### 3. **Personalize Marketing**
- Tailor offers based on route popularity
- Customize messaging by booking origin
- Optimize timing based on purchase lead patterns

### 4. **Resource Allocation**
- Focus sales team on high-probability leads
- Reduce spend on low-probability segments
- Shift budget to proven markets
---

##  Technical Stack

- **Language:** Python 3.12
- **Environment:** Google Colab (GPU-enabled)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Notebook:** Jupyter

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments:

- British Airways for the dataset
- Scikit-learn documentation and community
- Forage for the virtual internship program
