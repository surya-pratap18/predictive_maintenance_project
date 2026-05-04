# 📦 Submission Package
## LogicVeda Capstone — Predictive Maintenance

### 👤 Student Details
- **Name**: Surya Pratap Mallick
- **Project**: Predictive Maintenance & RUL Forecasting
- **Date**: May 2026
- **Project Code**: lv2-2026-03-01

---

### 🔗 Submission Links

| Deliverable | Link |
|-------------|------|
| 📊 Live Demo | https://logicveda-predictive-maintenance-project.streamlit.app |
| 💻 GitHub Repo | https://github.com/surya-pratap18/predictive_maintenance_project |
| 🎥 Demo Video | https://www.loom.com/share/6a3ac05c369a47a9bff10126bc0126f6 |
| 📄 PDF Report | https://drive.google.com/file/d/138vu0tKhtpmJKP1F9CRy9JtWZMx-nYb-/view?usp=drivesdk |

---

### ✅ Deliverables Checklist

- [x] Project Documentation (PDF)
- [x] Live Demo URL (Streamlit Cloud)
- [x] GitHub Repository
- [x] Demo Video (5-7 min)
- [x] Source Code & Notebooks
- [x] Model Artifacts

---

### 📊 Project Summary

| Week | Focus | Status |
|------|-------|--------|
| Week 1 | Data + Anomaly Detection | ✅ |
| Week 2 | RUL + Scheduler | ✅ |
| Week 3 | Dashboard + Monitoring | ✅ |
| Week 4 | Deployment + Polish | ✅ |

---

### 🤖 Model Performance

| Metric | Value |
|--------|-------|
| Anomaly F1 | > 0.85 |
| RUL MAPE | < 15% |
| RUL R² | > 0.90 |
| Scheduler solve time | < 30s |
| Tests passing | 6/6 |

---

### 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest tests/ -v

# Run drift detection
python src/drift_detection.py

# Run retraining pipeline
python src/auto_retraining.py