# Tactix

```

```
Tactix
├─ LICENSE
├─ README.md
├─ assets
│  ├─ output
│  │  ├─ Final_Result.mp4
│  │  └─ InterGoalClip_out.mp4
│  ├─ pitch_bg.png
│  ├─ samples
│  │  ├─ 20251122_Sunderland_Arsenal_goal_2.mp4
│  │  └─ InterGoalClip.mp4
│  └─ weights
├─ calibrate.py
├─ configs
│  ├─ football-pitch.yaml
│  └─ settings.yaml
├─ football-pitch.yaml
├─ main.py
├─ notebooks
│  ├─ demo_passing_net.ipynb
│  └─ test.ipynb
├─ requirements.txt
├─ runs
│  ├─ detect
│  └─ pose
│     └─ runs
│        └─ pitch_calibration
│           ├─ v1_n_27pts
│           │  ├─ args.yaml
│           │  └─ weights
│           ├─ v1_n_27pts2
│           │  ├─ args.yaml
│           │  └─ weights
│           └─ v1_n_27pts3
│              ├─ args.yaml
│              ├─ labels.jpg
│              ├─ results.csv
│              ├─ train_batch0.jpg
│              ├─ train_batch1.jpg
│              ├─ train_batch2.jpg
│              └─ weights
├─ setup.py
├─ tactix
│  ├─ __init__.py
│  ├─ config.py
│  ├─ core
│  │  ├─ __init__.py
│  │  ├─ geometry.py
│  │  ├─ keypoints.py
│  │  └─ types.py
│  ├─ engine
│  │  └─ system.py
│  ├─ semantics
│  │  ├─ __init__.py
│  │  └─ team.py
│  ├─ tactics
│  │  ├─ __init__.py
│  │  └─ pass_network.py
│  ├─ utils
│  │  ├─ __init__.py
│  │  ├─ generate_pitch.py
│  │  └─ video_io.py
│  ├─ vision
│  │  ├─ __init__.py
│  │  ├─ camera.py
│  │  ├─ detector.py
│  │  ├─ pose.py
│  │  ├─ tracker.py
│  │  └─ transformer.py
│  └─ visualization
│     ├─ __init__.py
│     └─ minimap.py
├─ tests
└─ tools
   ├─ convert_to_yolo.py
   ├─ download_data.py
   ├─ run_training.py
   ├─ train_models.py
   ├─ train_pitch.py
   └─ unzip_data.py

```