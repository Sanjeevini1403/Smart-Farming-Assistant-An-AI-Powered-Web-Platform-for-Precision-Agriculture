# Fixes applied before conference submission

These changes wire the real trained models into the live Flask app (backend/app.py)
and translations.py, replacing mock/placeholder logic that the paper described but
the code wasn't actually using.

## 1. Disease detection — was fake, now real
`/plant/` used to pick a result from a hardcoded 6-item list using
`hash(filename) % 6` — it never looked at the image. Now it loads
`ml_models/disease_mobilenet.h5` + `ml_models/class_labels.json` once at
startup and runs real MobileNetV2 inference (`predict_disease_ml()`).
Real classes: **Early Blight, Late Blight, Healthy** (tomato only — not "38+
diseases" as the old UI text claimed; that text is now fixed too).

## 2. Crop recommendation — was rule-based, now real Random Forest
`get_crop_recommendation_real()` (despite its name) was a hand-written
min/max range scorer, never touching `crop_model.pkl`. Now `predict_crop_ml()`
loads the real trained RandomForestClassifier + LabelEncoder and calls
`predict()` / `predict_proba()`. This model needs a `rainfall` feature that
the old form didn't collect — added to the Soil Analysis form.

**Known limitation carried over:** the shipped `crop_model.pkl` was trained
on a small local subset covering only 6 crops (chickpea, kidneybeans, maize,
mothbeans, pigeonpeas, rice), not the full 22-crop Kaggle dataset. Retrain
with the full dataset (see `evaluate_crop_model.py` docstring for how to get
it) before relying on the broader crop list described in the paper.

## 3. AI chatbot — was English-only keyword matching, now Gemini-backed
`get_ai_response()` now calls the Gemini API first (via `GEMINI_API_KEY` env
var) with a language-aware prompt, falling back to the original rule-based
responder (`get_ai_response_rulebased()`) if no key is set or the call fails.
A language dropdown (English / Tamil / Hindi) was added to the chat UI.

## 4. Hindi language support — was missing
`translations.py` only had `en` and `ta`. Added a full `hi` block.

## Still to do (needs your machine — network + TensorFlow)
1. Run `backend/evaluate_disease_model.py` and `backend/evaluate_crop_model.py`
   to get real accuracy/precision/recall/F1/confusion-matrix numbers for the
   paper. Do not hand-type numbers without running these.
2. Retrain `crop_model.pkl` on the full 22-crop Kaggle dataset for broader
   coverage (current model only knows 6 crops).
3. Set `GEMINI_API_KEY` in your environment for the chatbot to use Gemini;
   otherwise it silently falls back to the rule-based responder.
4. References [10], [12], [17] in the paper still need their full author
   lists verified against IEEE Xplore before camera-ready submission.
