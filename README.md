# SignSpeak AI — Sign Language to Text/Speech Hackathon Prototype

A Python prototype with:
- Streamlit UI
- Webcam hand landmarks with MediaPipe
- A starter Random Forest classifier trained on your own labeled landmark samples
- Text/sentence output
- Browser-based speech synthesis
- Demo mode, clearly marked as simulated input

## Important scope note
This starter is **not** a general sign-language translator. It recognizes only the labels represented in the data used to train it. Static hand-pose landmarks alone may not distinguish signs that depend on movement, facial expression, body posture, or context. For a serious Indian Sign Language (ISL) project, use an appropriately licensed ISL dataset and model temporal sequences where needed. Do not present demo mode as real recognition.

## Windows 11 setup

Install Python 3.11 (recommended for compatibility with the computer-vision packages). During installation, tick **Add Python to PATH**.

Open Command Prompt in this folder and run:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 from python.org first.

## Start the app

```bash
streamlit run app.py
```

The browser opens the interface. Use **Demo mode** to show the layout, sentence builder and speech output. Use **Live camera** to view hand landmarks.

## Train the model on your own samples

1. Close the Streamlit app if needed.
2. In the same activated terminal, run:

```bash
python collect_data.py
```

3. In the collector window, use:
   - `1` HELLO
   - `2` THANK YOU
   - `3` YES
   - `4` NO
   - `5` HELP
   - `6` WATER
   - `7` PLEASE
   - `8` SORRY
   - `SPACE` save one sample for the selected label
   - `Q` quit
4. Collect many examples per label, with varied hand positions and lighting. For a quick technical test, collect at least 30–50 samples per label; more and more varied data is better. Only label gestures you can perform consistently and define clearly.
5. Train:

```bash
python train_model.py
```

6. Restart Streamlit so it reloads `sign_model.joblib`, then select **Live camera**.

## How to demo honestly

- Show Demo mode first to explain the interface.
- For real recognition, show Live camera only after training.
- State which signs were trained and how many examples were collected.
- Report measured test results, not invented accuracy.
- Test with a separate person or recording session if possible. A random split of nearby frames can give misleadingly high accuracy.

## Suggested hackathon pitch

"SignSpeak AI is an assistive communication prototype that converts a limited set of recognized hand signs into text and speech. It combines computer vision, hand-landmark extraction, machine learning, and text-to-speech. Our current prototype demonstrates the end-to-end pipeline; our next milestone is a larger ISL-specific dataset and sequence-based recognition for moving signs."

## Troubleshooting

- **Camera unavailable:** allow camera access in Windows Settings > Privacy & security > Camera and close other apps using the camera.
- **No hand detected:** improve lighting, keep the full hand visible, and move the background clutter away.
- **Bad predictions:** collect more balanced samples; use consistent labels; test with a different person; consider normalized landmarks and sequence models.
- **Speech does not play:** click Speak text and check browser sound/permissions. Speech voices depend on the browser and operating system.
- **Dependency install fails:** use Python 3.11 in a fresh virtual environment and install the listed requirements.
