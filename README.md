# Online Meetings

## Requirements

install all used libs:

```
pip3 install -r requirements.txt
```

note that for dlib, you'll have to install some extra dependencies. For more, check out:
https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/

## Main

Opens a stats app and the window to the video

```
python3 main.py path_to_video --vis_inter 0.0 --plot_inter 0.1
```

To interrupt the analysis, click on the running video and press 'q'. This will open the analysis window.


#### Arguments:

- path_to_video (Default is '0', which will access the computer camera)
- vis_inter: set the visual analysis interval (Attention measuring, CNN classification)
- plot_inter: set the plotting analysis interval (Plotting in real-time attention window)

Note that there is a performance-analysis depth tradeoff. To analyze more that one person, be sure to set a higher vis_inter and plot_inter.

## Plot

Opens the statistical analysis of previously saved meetings

```
python3 plot.py name_of_csv_file
```

Arguments:

- name_of_csv_file: Has to be saved in the '../Saves' Folder. Csv File will be plotted on screen

## Construction

the main file is 'main.py'
to load saved analysis plots call 'plot.py'

- online_meetings
    - classes (contains classes for the attention model, plotting and the CNN)
    - datasets (folder to store the dataset for CNN training)
    - Saves (folder to store the analysis of earlier meetings)
    - Online Meetings (folder to store resources for the paper)
    - network_weights (folder to load the network weights from)

screen_to_tensor.py is LEGACY

#### Attention-Model:
	
- Constructs facial landmarks per face.
- Measures the attention by looking at the opening ratio of the eye and the gaze
- EAR: Eye-Aspect-Ratio to measure the opening of the eye
- GAZE: class stored in classes/gaze_tracking


## Visualization

- plotting.py - PyQt5 real time application for statistical plots
- mainwindow.ui - styling for application
- Feedback.py - statistical video summary

