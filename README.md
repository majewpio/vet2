# VetEye AI CRM Lead Scoring Demo

**Opis:**
Demonstrator inteligentnego CRM Lead Scoringu dla VetEye. Aplikacja Streamlit pokazuje:
- Top 20 leadów z ich score (zaokrąglonym do 2 miejsc po przecinku)
- Sortowanie i filtrowanie po cechach i rekomendacjach
- Szczegóły każdego leada wraz z wykresem ważności cech (SHAP)
- Rekomendacje skryptu rozmowy dostosowane do dominującej cechy leadu
- Statystyki POC: ROC-AUC, dystrybucja scoringu, liczba leadów w segmentach
- Mock integracji z CRM: „Wyślij do CRM” pokazuje payload JSON

## Struktura repozytorium

```
VetEye_CRM_AI/
├── assets/
│   ├── veteye_logo.png
│   └── demo_device.jpg
├── data/
│   └── leady_veteye_demo.csv
├── streamlit_app.py
├── requirements.txt
└── README.md
```
