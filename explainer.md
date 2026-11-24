# Image Sentiment Analysis — full explanation

This project looks at a picture and guesses whether its mood is **Positive** or **Negative**. It is a small deep-learning demo with two ways to use it: a **Streamlit** page and a **Flask** web page. The heavy work is already done — a trained model file is included, so you do not need to train anything to run the apps.

## What problem it solves

People post millions of images online. A system that can read the "feel" of an image can help with things like product reviews, social media monitoring, or sorting galleries. Here the job is kept simple: two answers only — positive or negative.

## Which dataset was used

Training used a CSV named **`image-Sentiment-polarity-DFE.csv`** from the Figure Eight / CrowdFlower "Data For Everyone" collection. It contains about **15,600 rows**. Each row has:

- `imageurl` — a link to a Flickr photo
- `which_of_these_sentiment_scores_does_the_above_image_fit_into_best` — a human-written sentiment label, for example `Positive`, `Negative`, or `Highly negative`

The CSV itself is not committed to this repo, but the training notebook shows it being uploaded in Google Colab and read with `pandas.read_csv`.

## How the dataset was prepared

In `ISA_Final.ipynb` / `isa_final.py`:

1. Only the sentiment column and the image URL column were kept, then renamed to `sentiment` and `imageurl`.
2. Rows were sorted by sentiment text.
3. A small balanced sample was made:
   - first **1,000** rows (positive side)
   - rows **13,000–14,000** (negative side)
   - combined into about **2,000** images
4. The rows were shuffled so the order could not bias training.
5. The sample was split into **1,000 training images** and **1,000 test images**.
6. Images were downloaded from their URLs with `imutils.url_to_image`, resized to **224 × 224**, and bad/dead links were skipped.
7. Labels were turned into numbers with `LabelEncoder`: `0 = Negative`, `1 = Positive`, then converted to one-hot vectors with `to_categorical`.

## How the model was made

The model uses **transfer learning** — instead of learning edges and shapes from scratch, it starts with VGG19 already trained on ImageNet.

Model build steps:

1. Load `VGG19` with `weights='imagenet'`, `include_top=False`, input shape `(224, 224, 3)`.
2. Freeze the VGG19 layers so they are not changed during training.
3. Add a custom head on top:
   - `GlobalAveragePooling2D`
   - `Dropout(0.5)`
   - `Dense(512, activation='relu')` with L1/L2 regularization
   - `Dense(2, activation='softmax')` for the two classes
4. Compile with `binary_crossentropy` loss, `Adam` optimizer (learning rate `0.001`), and accuracy metric.
5. Train for **64 epochs**, batch size **5**, with `validation_split=0.25`.
6. Evaluate on the test images, then save the trained model as **`best_model.h5`**.

`Completed_model.joblib` is another saved model artifact in the repo; the two apps use `best_model.h5`.

## How prediction works now

When you upload a picture:

1. `PIL` opens the uploaded file and converts it to a NumPy array.
2. OpenCV resizes it to **224 × 224** using `INTER_AREA`.
3. A batch of one image is passed to the loaded model.
4. `model.predict()` returns two probabilities.
5. `np.argmax()` picks the winning class:
   - `0` → **Negative**
   - `1` → **Positive**
6. The Flask page also shows the confidence as `probability × 10`, displayed as a score out of 10.

## Files in this project

- `main.py` — Streamlit app: upload an image, see the image and the prediction.
- `flask_app/flask_app.py` — Flask app with `/` (upload form) and `/predict` (POST image → result).
- `flask_app/templates/index.html` — styled upload page with drag-and-drop and preview.
- `flask_app/templates/output.html` — styled result page with badge and confidence bar.
- `isa_final.py` — exported training script from the notebook.
- `ISA_Final.ipynb` — original Colab notebook with outputs.
- `best_model.h5` — trained model used by both apps.
- `Completed_model.joblib` — extra saved model artifact.
- `Images/` — a few sample JPEGs used for trying predictions.
- `requirements.txt` — Python packages with versions that work on Python 3.11.
- `explainer.md` — this file.
- `agents.md` — short notes for future coding agents.

## Run it locally

The system Python on this Mac is 3.15, which cannot run TensorFlow 2.15, so a local `.venv` with Python 3.11 is used. It is already installed. If you ever need to recreate it:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

From the project root, run the Streamlit app:

```bash
.venv/bin/streamlit run main.py
```

Open the URL printed in the terminal, usually http://localhost:8501.

Run the Flask app:

```bash
(cd flask_app && ../.venv/bin/flask --app flask_app.py run --port 5001)
```

Open http://127.0.0.1:5001. It must be started from `flask_app/` because the code loads `../best_model.h5` relative to the working directory. Port 5001 is used because macOS often occupies 5000.

Quick model check:

```bash
.venv/bin/python -c "import tensorflow as tf; print(tf.keras.models.load_model('best_model.h5').input_shape)"
```

## Push changes to GitHub

```bash
git status --short
git add agents.md explainer.md
env GIT_AUTHOR_NAME='Amandeep Singh Jadhav' \
  GIT_AUTHOR_EMAIL='196829793+CodeMaster-AJ@users.noreply.github.com' \
  GIT_COMMITTER_NAME='Amandeep Singh Jadhav' \
  GIT_COMMITTER_EMAIL='196829793+CodeMaster-AJ@users.noreply.github.com' \
  git commit -m "Document project setup and training"
git push
```

## Good to know

- You only need to upload an image; no retraining is needed.
- Some original Flickr URLs in the dataset may no longer work, which is why the training code skips broken links.
- OpenCV reads images in BGR order while VGG19 was trained on RGB; this demo keeps the original behavior.
- This is a learning demo, not a production sentiment service.
- The two model files are around 80 MB each, so GitHub shows a large-file warning.
