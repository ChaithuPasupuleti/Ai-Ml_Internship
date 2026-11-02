## 🚀 Day 3: Data Loading, Cleaning & Inspection

### **Overview**
Day 3 focuses on robust data ingestion, schema validation, missing value sanitization, and automated unit testing adhering to MLOps best practices (PEP-8 formatting, linting, and safe path guards).

---

### **📁 Core Modules & Files**
* `data_loader.py` - Core script responsible for robust data ingestion, fallback paths, and schema validation.
* `data_cleaning.py` - Handles missing value treatment, null dropping, and dataset sanitization with empty-DataFrame guards.
* `tests/test_data_loader.py` - Unit test suite utilizing `pytest` and `tmp_path` fixtures to validate data loader and cleaner functions.
* `LICENSE` - Open-source MIT License for project distribution.

---

### **⚡ Quick Start & Execution Guide**

1. **Install Dependencies:**
   Ensure your virtual environment is active and required packages are installed:
   ```bash
   pip install -r requirements.txt