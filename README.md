# Supplier → TOW Mapper (develop)

Map supplier invoice product codes to internal **TOW** codes using a **persistent PostgreSQL (Neon)** database.  
All inserts/updates survive Streamlit restarts.

---

## 🔧 Setup (local)

```cmd
cd C:\Users\mkozi\OneDrive\Desktop\tow_mapper_develop
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
"# tow_mapper_develop" 
