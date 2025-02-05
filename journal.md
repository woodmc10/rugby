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
                - other tasks?