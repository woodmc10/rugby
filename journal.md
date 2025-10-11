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

### 3-6-2025
I've labeled 8 full games with light and dark tackles. 

Next Steps:
* Create a train/val split of the new larger dataset
* Run the model with a small number of epochs (3) to get baseline stats
* Evaluate possible hyperparameter changes and number of epochs for improved accuracy
* Evaluate the code for video clipping
    - I think it needs to be updated to clip variable lengths
* Tag some "general action" labels to see if the model can pick tackles out from continuous play

### 3-7-2025
Train/val split created by setting the USD v Loyola game as the validation set. This game is from a different stream than the other games and should provide a challenge that is comparable to "production."

Video clip length: I attempted to edit the code to handle video clips of various lengths. I discovered that padding would be required because the dataset batches require that all clips have the same shape (same number of frames). I thought about the situtation some more and realized that using this model for a full game would require feeding consistent length clips through the model and having them tagged as action/non-action and then classifying the action clips. In this situation, all clips would be of a consistent length. With this knowledge I decided to update the code to take the last X frames from the tagged regions instead of clipping the region into multiple sets of X frames. Most of the tackles I tagged started when contact was initiated and ended when the ball carrier went to ground. The key part of the tackle is the ball carrier going to ground, so taking the last X frames from every tag should provide a set of clips with a successful tackle in each clip. 
- The runtime was crasshing when using clip length of 80 frames, but succeeded with a clip length of 40 frames

Basline stats: ~20% accuracy (all predictions for one class)

Next Steps (week of 3/10)
* Write code to save model weights so training doesn't have to happen every time the notebook runs
* Inspect predictions (labels and viewing clips)
* Inspect training 
    - How balanced are the labels?
    - Is 40 frames enough? Are there clips with no tackle?
* Evaluate possible hyperparameter changes and number of epochs for improved accuracy
* Tag some "general action" labels to see if the model can pick tackles out from continuous play

### 3-10-2025
I had some trouble getting the model to save. I started with some code from Claude, but it didn't work. I wound up needing to save the model weights, recreate the model architecture in a new notebook and then load the saved weights. The model architecture still needs to be loaded with a frozen backbone to keep the model the same (and to avoid shape mismatch errors). 

### 3-18-2025
The model is struggling to learn with the small amount of data provided. I will continue to work on labeling more data, but the fastest way to get more data for training will be to switch from "white tackle" vs "dark tackle" to "tackle" vs "no tackle." I double the amount of training data I have without having to get permission to use more videos by just adding some game segments without tackles. 

### 4-11-25
I've finished labeling all the videos and downloaded the json files. Today I need to save the files in the correct place, zip them and store the zip in Google Drive, and update the code to treat white_tackle and dark_tackle as a single "tackle" category. After that, I should be able to train a new tackle vs no-tackle model and see if it performs better than the white vs dark tackle model.

I upated the code to handle the new scenario, but the model did not improve. I need to dig into what's happening.

### 5-5-2025
I got permission to use the Bowdoin Women's Rugby matches that I've been coding for training the model. I'll label all of their 2025 matches and add them to the training. I expect I'll need to do more investigating to determine why the model is not working, but it is possible the increase in data will improve the model and the extra data will be necessary for the final model anyway.

Other Ideas (before I start on the labeling):
- The model may need lots more data, or it may be better suited to training with lots of different labels. To check this, I'd like to take the original notebook examples and reduce the number of classes down to two.

### 7-1-25
I've taken a step back from this due to the complexity of the models. I'd like to jump back in but it will be at least two weeks before I can get started due to the move to Iowa. Here is my plan for recommitting one day a week to this project.

7/15: 
    1. Troubleshoot MoviNet original notebook - reduce to two classes
    2. Decide on a next approach - how many classes should I label?
        Options:
        * stick with only tackle
        * pick more classes to try to label every moment of a game
7/22: 
    1. Label one Bowdoin video
    2. Label three HSBC 7s games
    3. Design a system to chunk the videos into segments for labeling
7/29:
    1. Label one HSBC 7s tournament
    2. Train a model with HSBC labels and compare it to the model trained with college game footage
    3. Reach out to Laura about fall season film plans
8/5:
    1. Explore models other than MoviNet2
    2. Label one Bowdoin video
    3. Plan August work


### 9-4-25
I created a Claude Project to help me work through the rugby tackles CV model because I haven't touched it in months. Most of the reason that I haven't worked with it is because I didn't know what to take on next. Having a Claude project will help me get some direction. I'm aware this may not be completely right, but LLMs are great for getting past the writer's block that I'm experiencing now. I'll use it for that, correct errors as I find them, and hope that it leads to more progress.

I started out to continue the investigation into why MoViNet was not making good predictions on my tackling dataset, but I quickly ran into issues with the CoLab notebooks. I decided to pivot and start the project fresh from AWS. 

### 9-14-25
Notes from AWS setup
- created a new IAM user (mw_rugby_tackles)
    * set up access
- created s3 buckets for data, models, and logs
    * `aws s3 ls`
- aws has suspended my ability to create EC instances, waiting for assistance

### 9-17-25
Docker Code was all written by Claude
The orginal build files didn't work and the conversation ran out of memory during troubleshooting. Claude tried a few solutions to the installation conflicts (blinker 1.4) but nothing was working. I found a different solution through a Google Search AI answer (`RUN pip install --ignore-installed -r requirements.txt`) and now have a successfully built Docker image.

Trying to test Docker set up with Claude command `docker-compose run rugby-dev python -c "import tensorflow as tf; print(tf.__version__)"` returned this warning:
**WARN[0000] /Users/marthawood/Code/rugby/aws/rugby-tackles-project/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion**
- This created a running container that I can do some minimal interaction with through the Docker Desktop, but I need a refresher on interacting with Docker. 
- The dockerdocs suggest using `docker compose watch` for interactive development, but that is a website development example. I need to figure out how to do my daily development with Docker.

### 9/18/25
Tensorflow can have a mismatch between the build architecture and the CPU architecture. This is likely the issue with my commands not running, but I haven't been able to figure out how to get them to run. Instead of spinning my wheels on tensorflow locally (which I probably will have computer limiations for anyway). I'm going to set up docker with different specifications for local and aws. I'll make sure the data processing is working locally and then move to aws for any large processing or training.

Docker built successfully and contains the necessary dependencies. Data doesn't seem to be located where it's expected, so tomorrow I'll check out where the data is being stored and ensure the processing file works. 

### 10/2/25
Data processing is executing fully. It is currently generating 3 different labels - no tackle, white tackle, and dark tackle. Next, I need to read through the CoLab notebooks and see what other processing is needed before training/fine tuning the model. Then, it will be time to get set up with AWS to do basic model training. Once I complete those steps, I need to make sure I've developed the appropriate structure/systems to log model and data experiments. 

**Claude Instructions on Docker Daily Interaction**:

```bash
# Starting The Day:
cd aws/rugby-tackles-project

# Run the data_processing.py file
docker-compose -f docker-compose.local.yml run --rm rugby-dev python src/data_processing.py

# Or start Jupyter Lab
# docker-compose up jupyter
# Then visit http://localhost:8888


# Running Existing Code:
# Test your dataset creation code
# docker-compose run --rm rugby-dev python src/dataset_cluad.py

# Run any script
# docker-compose run --rm rugby-dev python scripts/train.py --config configs/base_config.yaml


# Stopping Containers:
# Stop Jupyter
docker-compose -f docker-compose.local.yml down

# Clean up stopped containers
# docker system prune
```

These will need to be updated for the new approach - having different docker files for local development and AWS development due to tensorflow configuration issues. Commented code has not been checked/updated

### 10/11/25
The data_processing.py file is working locally to generate the clips and the dataframe for training and validation annotations. Next steps are to get set up with AWS and get a MoViNet model fine tuned. 