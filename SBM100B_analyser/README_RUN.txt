SBM100B Analyzer

Run on Mac:
1) Open Terminal in this folder.
2) Install dependencies if needed:
   python3 -m pip install -r requirements.txt
3) Run:
   python3 app/main.py


Project structure:
- app/main.py              
- core/analysis.py         analysis logic, unchanged from uploaded file
- core/audio_io.py         audio device helper, unchanged
- core/calibration.py      calibration helper, unchanged
- recordings/             saved WAV files
- results/plots/           exported plots
- results/tables/          exported CSV tables
- config/                  calibration file
