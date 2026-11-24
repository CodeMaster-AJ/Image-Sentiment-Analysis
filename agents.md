# Agent notes

- Repository root: `Image-Sentiment-Analysis`. Use `.venv/bin/python`, `.venv/bin/streamlit`, and `.venv/bin/flask`; the system Python is 3.15 and cannot run TensorFlow 2.15.
- Do not commit `.venv/` or `flask_app/__pycache__/`.
- Streamlit entrypoint is `main.py`. Flask entrypoint is `flask_app/flask_app.py`.
- Flask loads `../best_model.h5` relative to its working directory, so run it from `flask_app/`.
- No lint or typecheck command is configured. Verify with:
  - `.venv/bin/python -c "import tensorflow as tf; tf.keras.models.load_model('best_model.h5')"`
  - starting Streamlit and Flask and checking `http://localhost:8501` and `http://127.0.0.1:5001`
- Keep the Flask form field name `image` and result variables `prediction` and `val` unless explicitly asked to change them.
- Git commits use `Amandeep Singh Jadhav <196829793+CodeMaster-AJ@users.noreply.github.com>` and `origin` points to the `CodeMaster-AJ` fork.
- Do not add code comments unless requested.
