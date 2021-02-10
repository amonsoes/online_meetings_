# Online Meetings

## Main

Opens a stats app and the window to the video

```
python3 main.py path_to_video --vis_inter 0.0 --plot_inter 0.1
```

Arguments:

- path_to_video          - Default is '0', which will access the computer camera
- vis_inter: set the visual analysis interval
- plot_inter: set the plotting analysis interval

## Plot

Opens the statistical analysis of previously saved meetings

```
python3 plot.py name_of_csv_file
```

Arguments:

- name_of_csv_file: Has to be saved in the '../Saves' Folder. Csv File will be plotted on screen

## Speech Model


## Visual Model

#### Attention-Model:
	
Constructs facial landmarks per face.
Measures the attention by looking at the opening ratio of the eye
by using EAR
EAR: Eye-Aspect-Ratio to measure the opening of the eye

-eye_detection.py


## Visualization

- plotting.py - PyQt5 real time application for statistical plots
- mainwindow.ui - styling for application
- Feedback.py - statistical video summary

