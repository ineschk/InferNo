# NSL-KDD Network Intrusion Scoring API

## Structure
```
nsl-kdd-app/
├── main.py            ← API FastAPI
├── train_model.py     ← Entraîne et sauvegarde model.pkl
├── requirements.txt
├── static/
│   └── index.html     ← Interface utilisateur
└── KDDTrain+.txt      ← Dataset (à télécharger)
```

## Installation

```bash
pip install -r requirements.txt
```

## Étapes

### 1. Télécharge le dataset
https://www.kaggle.com/datasets/hassan06/nslkdd
→ Place `KDDTrain+.txt` dans ce dossier.

### 2. Entraîne le modèle
```bash
python train_model.py
```
→ Génère `model.pkl`

### 3. Lance l'API
```bash
uvicorn main:app --reload --port 8000
```

### 4. Ouvre l'UI
http://localhost:8000

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/v1/health` | Statut de l'API |
| POST | `/v1/score` | Score une connexion (41 features) |
| POST | `/v1/batch` | Score plusieurs connexions |

## Exemple

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"features": [0,"tcp","http","SF",181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0,0,0,0,1,0,0,9,9,1,0,0.11,0,0,0,0,0]}'
```

Réponse :
```json
{
  "score": 12,
  "label": "normal",
  "confidence": 0.98,
  "model_version": "1.0.0"
}
```
