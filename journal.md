# 2025
## January
I've picked up this project again with a focus on labeling tackling and missed tackles. This switch in priority comes after some conversations with women's college team coaches about how they use data and what gaps need to be filled. The current automated solutions for other sports like basketball (Hudl) and soccer (Veo) do not work as well for rugby. One of the tasks they are intersted in offloading is breaking down game video to collect stats. Some of the primary stats are tackles and missed tackles.

### 1-23-25
I'm starting off the project with some simple tutorials and finding a solution for data labeling.
* Tutorials
    - https://blog.tensorflow.org/2023/01/using-tensorflow-for-deep-learning-on-video-data.html
    - These tutorials show how to use tensorflow and keras to load, predict, and fine tune the Movinet models. The colab tutorials are slightly outdated and the most recent packages were causing errors. I've saved the edits I made to the final tutorial into this repo so that I can reference it later.
* Data Labeling Options
    - Datature Nexus
        * This platform seemed like a great option with integrated data labeling and model training. Unfortunately the project would not complete the upload of the first video
    - CVAT
        * This labeling software seems to be more targetted at image labeling with solutions for boxes, maps, etc. I could not find an easy way to label a section of a video with a particualar action. But it did allow me to upload the video with the highest resolution because it did not have a 100Mb limit. The video playback on the annotation tab was slow with the image buffereing every 30ish frames (about every second).
    - Supervisely
        * Of the three options that I tried I like this interface the best for labeling segments of frames for action recognition. But it does have a 100Mb limit for uploading videos that may become a limit with longer videos. 
* Downloading frames from YouTube
    - https://www.4kdownload.com/howto/how-to-extract-frames-from-a-youtube-video/2


### 2-3-25
Current Status: I've walked through the Google Colab tutorials and used Supervisely to label tackles in a single game. I plan to spend this week ensuring I can actually run a model on game film. 
* Questions/Concerns:
    - The tutorials were all designed to evaluate a complete, short video and label it with a single action. Game film will not be a single action, the whole film needs to be broken down and multiple actions extracted from it. I need to develop a process for completing this preprocessing step.
        * Article about temporal action location: https://blog.ml6.eu/sports-video-analysis-in-the-real-world-realtime-tennis-action-recognition-using-movinet-stream-813200aa589f
            - temporal pooling
                * Either take a segment around each frame and classify it so that each frame in the video receives a classification - slow
                * Or take equal length segments from the video and classify each - problematic when actions are not equal in length (scrum v tackle)
            - action proposal generator
                * predicts the segments of a video that are highly likely to include actions
* Research Notes:
    - definitions of rugby actions and evaluations
        * https://bjsm.bmj.com/content/54/10/566
        * research paper - incredibly detailed in the options/definitions of the actions
            - these may not be relevant for Laura, but I'm keeping the reference in case I need to generate a definition for an action in labeling
    - Terms:
        * Scene Classification
        * Temporal Action Localization
            - Temporal Action Detection
                * AdaTAD: https://arxiv.org/pdf/2311.17241
                    - Model architecture is limited. The intro discusses increases of frame input to 1500. The video I have of a single game is closer to 35,000 frames. 
                * TadTR: https://arxiv.org/pdf/2106.10271
                    - Temporal action detection using transformers
                * Helpful GitHub: https://github.com/sming256/openTAD?tab=readme-ov-file
        * Action Detection
    - I'm going to focus on Temporal Action Localization/Detection methods

### 2-5-2025
* movinet_tutorial.ipynb
    - Tutorial uses UCF-101 dataset to fine tune the model.
    - This needs to be replaced with a rugby dataset.
        * Steps:
            1. Understand dataset generation with tensorflow
                - https://www.tensorflow.org/datasets/add_dataset
            1. Figure out how to create a dataset from the Supervisely labeled rugby videos
                - clip videos
                - save labels in correct location/format

### 2-12-2025
I've used Claude to help me create a python file that can use the annotations file from Supervisely to chunk out labeled images and create a Tensorflow dataset. 
* Goal: use the dataset to finetune movinet - no expectation of performance, just confirming the steps can be done.
    - Subtask: Complete documentation in a clear way that Future-Me could run if I didn't work on this project for the next year.
* I tried to set up DVC, but it's having errors with the Google Drive authentication and I don't want to set up AWS or another storage system. For now, the DVC infrastructure is present but unused.

### 2-17-2025
Colab tutorial is crashing when trying to use TPU for inference. I haven't been able to identify if this is a versioning problem. For now I'm going to try to move forward with fine tuning on the tiny dataset on a CPU. I'll need to investigate other ways of training/inference if the dataset works.
- I found a work around for the error. Using the CPU allows for inference. Now I need to break out the parts that need GPU for training. Likely will be best to create a separate training notebook.

### 2-19-2025
- I created some very basic documentation to remind myself of the steps I'm following. This will be helpful if I need to step away from the project for a few weeks, but much clearer documentation will be needed to execute the workflow if I step away for a long time.
- I split the movinet tutorial notebook into two separate notebooks. One for the building and running the movinet model from tensorflow hub, and one for fine tuning. The movinet predictions encounter an error when running on the TPU, but the TPU is necessary for fine tuning. Instead of switching runtimes halfway through the notebook, I separated the notebook at the point where the runtime needs to be switched.

#### Notes on Current State
1. Model Options - I'm sticking with the MoViNet models for now because of the availability of the Colab tutorials. These models will classify a video clip, but will not idenfity a section of interest from a longer video. 
    - Detecting clips for classification will be a task for later.
    - I'm interested in exploring the [OpenTAD GitHub](https://github.com/sming256/openTAD?tab=readme-ov-file) more.
1. Data Labeling - Supervisely will be used for data labeling from the internet. The reasoning behind that decision can be found above. 
    - I still need to establish the best way to extract training clips from Veo so that I don't need to repeat the labeling tasks after uploading the game film to Supervisely.
1. Tutorial Updates - The MoViNet tutorials themselves required some updates due to outdated libraries and CPU/GPU conflicts. I split the beginning of the tutorial (building and running) off from the fine tuning section of the tutorial. I had to install an older version of matplotlib to handle the visualization of the GIFs and plots, and had to use the legacy Keras integration with Tensorflow. The build and run notebook should be run using the CPU hardware, and the fine tuning notebook should be run using the GPU hardware. I added code to mount GDrive and clip training videos from full game film that has already been labeled using Supervisely. 
    - The test set is currently the same as the training set. This is terrible practice for truly evaluating model performance, but I'm using it just to verify the code can be executed. This needs to be corrected ASAP.
    - The MoViNet notebooks tend to break, I should run them regularly to ensure I've captured and corrected all the errors.


#### Next Steps
1. Schedule more conversations with Laura.
    - Get familiar with the terminology
    - Understand how coaches review film and what they want to see clipped
1. Develop a tagging system
    - Classes and sub classes
    - Assigning events to teams
    - Assigning events to players
1. Continue labeling and training
    - Get a test set that isn't a duplicate of the training set!!!
    - Label more games
        * Confirm with Laura that I can use their games for model training
        * Get clips from Veo/Hudl for training
        * Label more games from the internet
1. Save model for inference
1. Work on full game solution

### 2-24-2025
Today I tried to download a YouTube video of multiple 7s games. I trimmed the original video down so that each game was a single file using Quick Time Player. When I tried to upload these to Supervisely I got an error about B-frames. A chat with Claude indicated that the error may be due to the way Quick Time Player trims and encodes the video.

Possible Solutions:
1. Supervisely: run an application called Transcode Videos. This is taking hours for 3 videos and has not finished yet.
1. Claude command line: reencode the image with ffmpeg
    - `ffmpeg -i input.mp4 -c:v libx264 -preset medium -crf 28 -bf 0 output.mp4`
    - This worked, but reencoding with the above settings increased the file size slightly. It will be important to pay attention to file size given the 100MB limit on Supervisely. 