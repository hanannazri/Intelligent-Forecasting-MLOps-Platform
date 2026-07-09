#  **Intelligent Retail Demand Forecasting & Inventory Intelligence Platform**

*Predict retail demand, optimize inventory decisions, and explore business insights through machine learning, interactive analytics, and an AI-powered assistant.*

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white"/>
<img src="https://img.shields.io/badge/Gemini_AI-4285F4?style=for-the-badge&logo=google&logoColor=white"/>

</p>

<div align="center">

<a href="https://intelligent-forecasting-mlops-platform.streamlit.app/">
  <img src="https://img.shields.io/badge/Open%20Live%20Dashboard-Click%20Here-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=0E1117"/>
</a>

</div>

---

## **Dashboard Preview**

<p align="center">
  <img src="assets/dashboard-overview.png" alt="Dashboard Overview" width="47%" />
  <img src="assets/product-recommend.png" alt="Product Recommendation" width="47%" />
</p>

<p align="center">
  <img src="assets/inventory.png" alt="Inventory Planner" width="47%" />
  <img src="assets/insights1.png" alt="Business Insights" width="47%" />
</p>

<p align="center">
  <img src="assets/ai-assistance.png" alt="AI Inventory Assistant" width="75%" />
</p>

---

## **Project Overview**

The **Intelligent Retail Demand Forecasting & Inventory Intelligence Platform** is a production-oriented analytics application that helps retailers forecast product demand, optimize inventory decisions, and explore business insights through an interactive dashboard and AI-powered assistant.

Built on the M5 Forecasting dataset, the platform combines data engineering, feature engineering, machine learning, inventory intelligence, experiment tracking, REST APIs, containerized deployment, and interactive analytics into a single business-focused solution.

---
## 📊 Project at a Glance

| Feature | Details |
|----------|----------|
| 📦 Products Analyzed | **1,437** |
| 📅 Forecast Horizon | **28 Days** |
| 🤖 Forecasting Model | **LightGBM** |
| 📈 Baseline Improvement | **12.96%** |
| 🎯 MAE | **1.204** |
| 🧠 AI Assistant Accuracy | **85%** |
| 🐳 Deployment | **Docker & Docker Compose** |
| 🔄 CI/CD | **GitHub Actions** |

---

## **Business Problem**

Retail businesses continuously face two expensive inventory challenges:

### **Understocking**

- Products become unavailable
- Customers leave without purchasing
- Sales opportunities are lost
- Customer satisfaction decreases

### **Overstocking**

- Excess capital is locked in inventory
- Warehousing costs increase
- Perishable products may expire
- Inventory turnover decreases

Making inventory decisions based solely on historical averages or manual planning often ignores important business signals such as pricing changes, seasonality, holidays, and purchasing trends.

This project addresses that problem by forecasting future product demand and translating those forecasts into practical inventory recommendations that support more informed replenishment decisions.

---

##  **Platform Features**

| Feature                   | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| 📈 Demand Forecasting     | Predicts future retail demand                   |
| 📦 Inventory Intelligence | Generates reorder recommendations               |
| 📊 Interactive Dashboard  | Visualizes KPIs and inventory insights          |
| 🤖 AI Assistant           | Answers inventory questions in natural language |
| 🌐 REST API               | Serves forecasting and inventory endpoints      |
| 📈 MLflow                 | Tracks experiments and model versions           |
| 🐳 Docker                 | Containerized deployment                        |
| 🔄 GitHub Actions         | Automated CI pipeline                           |



---

## **Tech Stack**

| Layer               | Tools                                         |
| ------------------- | --------------------------------------------- |
| Programming         | Python                                        |
| Data Processing     | Pandas, NumPy, DuckDB, Parquet                |
| Machine Learning    | LightGBM, XGBoost, Scikit-learn               |
| Experiment Tracking | MLflow                                        |
| API                 | Flask                                         |
| Dashboard           | Streamlit, Plotly                             |
| AI Assistant        | Gemini API, Rule-Based Routing, Tool Registry |
| Deployment          | Docker, Docker Compose                        |
| CI/CD               | GitHub Actions                                |
| Version Control     | Git, GitHub                                   |

---

## **Project Architecture**

The platform is designed as a modular, production-oriented machine learning system where each component has a single responsibility. This separation improves maintainability, scalability, and deployment flexibility.

```mermaid
flowchart TD

A[Raw M5 Dataset]
--> B[Data Processing]
--> C[Feature Engineering]
--> D[LightGBM Model]

D --> E[MLflow Tracking]
D --> F[28-Day Demand Forecast]

F --> G[Inventory Recommendation Engine]

G --> H[Business Reports]

H --> I[Streamlit Dashboard]
H --> J[AI Assistant]

J --> K[Rule-Based Router + Gemini API]

L[Flask REST API]
--> M[External Applications]

N[Docker]
--> I

N
--> L

O[GitHub Actions]
--> N
```
---


<details>
<summary><b>📂 Repository Structure</b></summary>

<br>

```text
Intelligent-Forecasting-MLOps-Platform/
│
├── .github/
│   └── workflows/              # GitHub Actions CI pipeline
│
├── data/
│   ├── raw/                    # Original M5 dataset
│   ├── processed/              # Cleaned datasets
│   └── features/               # Engineered features
│
├── models/                     # Trained model artifacts
│
├── reports/                    # Forecasts & inventory recommendations
│
├── src/
│   ├── ai_assistant/           # AI assistant & routing logic
│   ├── api/                    # Flask REST API
│   ├── business/               # Inventory recommendation engine
│   ├── dashboard/              # Streamlit dashboard components
│   ├── data/                   # Data ingestion & preprocessing
│   ├── features/               # Feature engineering
│   └── models/                 # Model training & forecasting
│
├── mlruns/                     # MLflow experiment tracking
│
├── Dockerfile                  # Dashboard container
├── docker-compose.yml          # Multi-container deployment
├── requirements.txt
└── streamlit_app.py            # Dashboard entry point
```

</details>

The repository follows a modular architecture where data processing, machine learning, APIs, dashboard components, and AI services are organized into independent modules.

---

## **End-to-End Machine Learning Pipeline**

The project follows a complete machine learning workflow from raw retail data to business decision support.

```mermaid
flowchart LR

A[Raw Sales Data]

B[Calendar Data]

C[Sell Prices]

A --> D[Data Validation]

B --> D

C --> D

D --> E[Data Transformation]

E --> F[Feature Engineering]

F --> G[LightGBM Training]

G --> H[MLflow]

G --> I[Future Forecast]

I --> J[Inventory Recommendation]

J --> K[Business Dashboard]

J --> L[AI Assistant]
```

---

## **Data Engineering Pipeline**

The forecasting pipeline begins by integrating historical sales, calendar events, and pricing data into a unified feature dataset.

The pipeline performs:

- Data ingestion
- Data validation
- Data transformation
- Feature dataset generation

The resulting dataset serves as the foundation for forecasting, inventory recommendations, dashboard reporting, and AI-assisted business queries.

---

## **Feature Engineering**

The forecasting model uses engineered features to capture historical demand patterns, seasonality, pricing behavior, and product-level characteristics.

Feature categories include:

- Historical demand features (lags and rolling statistics)
- Calendar and seasonal features
- Pricing and price change features
- Product, department, and category aggregations

These features help the model learn both short-term demand fluctuations and long-term purchasing trends.

---

## **Forecasting & Inventory Intelligence**

The platform forecasts retail demand using a supervised machine learning model and transforms those predictions into actionable inventory recommendations.

### **Forecasting**

The final production model was selected after comparing multiple forecasting approaches against a baseline model.

| Model | Purpose |
|--------|---------|
| Baseline | Performance benchmark |
| XGBoost | Model comparison |
| **LightGBM** | Final production model |

### **Inventory Intelligence**

Forecasts are converted into business-ready recommendations, including:

- Recommended order quantity
- Reorder point
- Target stock level
- Demand risk category

This business layer bridges the gap between machine learning predictions and operational inventory planning.

---

## **MLflow Experiment Tracking**

MLflow is used to track model training experiments, including parameters, evaluation metrics, and model artifacts. This ensures that experiments remain reproducible and makes it easier to compare different forecasting models throughout development.

---

## **AI Business Assistant**

The platform includes an AI-powered assistant that enables users to query inventory insights using natural language.

To ensure reliable business responses, the assistant retrieves verified data through predefined backend functions instead of generating inventory values directly.

### **Supported Business Questions**

- Inventory summary
- Products requiring replenishment
- High-risk inventory items
- Product-level reorder recommendations

### **Example Queries:**

- Which products require immediate replenishment?
- Show the inventory summary.
- What is the reorder recommendation for this product?

### **Assistant Architecture**

```mermaid
flowchart LR

A[Business Question]

A --> B[Rule-Based Router]

B -->|Matched| C[Approved Backend Function]

B -->|Fallback| D[Gemini AI]

D --> C

C --> E[Inventory Recommendation Dataset]

E --> F[Verified Business Result]

F --> G[Natural Language Response]
```

> High-confidence inventory queries are processed locally through deterministic routing. Gemini is used only when additional language understanding is required.

### **Design Benefits**

- Faster responses
- Reduced API usage
- Lower operating cost
- Reliable business outputs
- Graceful fallback during API quota limits

---

## **REST API Layer**

The platform exposes forecasting and inventory capabilities through a lightweight Flask REST API, enabling integration with external applications and business workflows.

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service health check |
| `GET /forecast/28days` | Retrieve 28-day demand forecasts |
| `GET /inventory/recommendations` | Retrieve inventory recommendations |

---

## **Streamlit Dashboard**

The Streamlit application serves as the primary business interface for interacting with forecasting results.

Instead of generating predictions every time the application loads, the dashboard reads precomputed forecast and inventory reports.

This batch inference approach significantly improves dashboard responsiveness while maintaining consistent business outputs.

Dashboard capabilities include:

- Executive KPIs
- Inventory Recommendation Table
- Department Filters
- Demand Risk Analysis
- AI Assistant
- Forecast Summary

The dashboard is intended for operational users who require business insights rather than direct access to machine learning code.

---

## **Continuous Integration**

GitHub Actions automatically validates the project whenever new code is pushed.

The workflow includes:

- Dependency installation
- Project validation
- Import checks
- Docker image build verification

This helps ensure that the project remains reproducible and deployable across environments.

---

## **Docker Deployment**

The project is fully containerized using Docker and Docker Compose, providing a reproducible environment for both the Streamlit dashboard and Flask API.

```mermaid
flowchart LR

A[Docker Compose]

A --> B[Streamlit Container]

A --> C[Flask API Container]

B --> D[Business Dashboard]

C --> E[Forecast & Inventory API]

D --> F[Business User]

E --> F
```

Start all services with:

```bash
docker compose up --build
```

| Service | Default Port |
|----------|-------------:|
| Streamlit Dashboard | 8501 |
| Flask REST API | 5000 |

---

## **Model Performance**

The forecasting pipeline was evaluated using a time-based validation strategy, where the most recent observations were held out as the test period.

The final production model was selected after comparing multiple forecasting approaches against a baseline model.

### **Final Evaluation**

| Metric | Result |
|----------|--------|
| Final Model | LightGBM |
| Forecast Horizon | 28 Days |
| Mean Absolute Error (MAE) | **1.204** |
| RMSSE | **0.667** |
| Improvement over Baseline | **12.96%** |

The trained model demonstrated consistent forecasting performance while maintaining efficient inference, making it suitable for downstream inventory planning.

---

## **Business Impact**

The objective of this project extends beyond producing accurate forecasts.

Forecasts are transformed into actionable inventory recommendations that can support operational decision-making.

The platform helps answer business questions such as:

- Which products require replenishment?
- Which products have the highest stockout risk?
- How much inventory should be ordered?
- Which departments require immediate attention?
- What inventory insights can be obtained through natural language?

By combining forecasting with inventory intelligence, the platform bridges the gap between machine learning predictions and business operations.

---

## **Project Results**

| Metric                    |                  Result |
| ------------------------- | ----------------------: |
| Forecast Horizon          |                 28 days |
| Products Forecasted       |                   1,437 |
| Dataset Size              |      ~2.75 million rows |
| Model Used                |                LightGBM |
| MAE                       |                   1.204 |
| RMSSE                     |                   0.667 |
| Improvement over Baseline |                  12.96% |
| AI Assistant Evaluation   | 20 / 20 correct intents |
| AI Assistant Accuracy     |                    100% |

---

## **Engineering Highlights**

The platform was designed with practical software engineering principles to improve performance, maintainability, and usability.

- **Modular architecture** – The project is organized into independent modules for data processing, forecasting, business logic, APIs, the dashboard, and the AI Assistant, making the codebase easier to maintain and extend.

- **Precomputed forecasting pipeline** – Forecasts are generated before the dashboard loads, reducing response times and providing a smoother user experience.

- **Business-focused recommendations** – Instead of displaying raw predictions, the system converts forecasts into reorder quantities, reorder points, and inventory recommendations that are directly useful for decision-making.

- **Hybrid AI Assistant** – High-confidence inventory questions are handled through deterministic routing, while Gemini is used for natural language understanding, balancing reliability with flexibility.

- **Reproducible experimentation** – MLflow tracks parameters, metrics, and model artifacts, making experiments easy to reproduce and compare.

- **Containerized deployment** – Docker and Docker Compose ensure that the dashboard and API can be deployed consistently across different environments.

---

## **Future Improvements**

Potential enhancements include:

- Forecast all stores and product categories
- Integrate real inventory databases
- Add automated model retraining
- Implement model monitoring and drift detection
- Store model artifacts in cloud object storage (AWS S3)
- Deploy the Flask API to a cloud environment
- Add authentication and role-based dashboard access
- Support real-time inventory updates
- Expand the AI Assistant with analytical capabilities and additional business workflows

---
### **Acknowledgements**

This project uses the **M5 Forecasting** dataset, one of the most widely used public benchmarks for retail demand forecasting research.

Special thanks to the open-source community and the developers of Python, LightGBM, Streamlit, Flask, MLflow, Docker, and GitHub Actions for making projects like this possible.

---

### **Connect With Me**

**Hanan**

Aspiring Data Scientist & ML Engineer

-  LinkedIn: https://linkedin.com/in/hanan-nazri
-  GitHub: https://github.com/hanannazri
