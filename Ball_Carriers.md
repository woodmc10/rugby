# Identify Ball Carriers in Rugby Highlights
### Subject Selection
Jumping into the problem of training a model to predict knock-ons from video of the Women's Six Nations seemed daunting. I haven't yet worked with videos, I haven't yet trained a custom computer vision model, and I don't yet know how to predict events that are difficult or impossible to identify in still images. Instead, I started with a much simpler problem. I will be trying to identify the ball carrier in the highlights of Women's Six Nations games. The idea for this step came from this [blog](https://blog.paperspace.com/yolov7/).

### Preparing the Data
#### England v Scotland Highlights
Highlights were chosen to reduce the amount of data used for the initial labeling and the videos are available on YouTube. The first step of training the model was to obtain images for labeling from the videos. 4K Video Downloader provides software to download YouTube videos. Frames from this video were extracted one frame per second using VLC. (I followed instructions in this [blog](https://www.4kdownload.com/howto/how-to-extract-frames-from-a-youtube-video/2).) 

### Labeling the Data with Roboflow
Roboflow is a computer vision platform that provides data labeling and management, and provides the ability to train and deploy models all in one place. The Roboflow labeling UI provides a simple, intuitive way to label all the uploaded frames from the highlight video. The frames were labeled with scotland_ball_carrier or england_ball_carrier when a ball carrier was present. 

#### Labeling Challenges
While working with the frames it became apparent that labeling ball carriers in a rugby game would be difficult. Due to the nature of the game ball carriers are often occluded from view, and knowledge of the flow of the game is needed to identify the ball carrier because the ball cannot been seen. 
##### Scrums
Before the scrum the scrum half will have possession of the ball. When the ball is fed into the scrum there is no ball carrier. Finally, the eight man or scrum half will take the ball from the scrum. When this occurs it is difficult to identify the moment when there is a ball carrier because the ball is often occluded. The timing of ball in and ball out was estimated during labeling.
##### Mauls
Many of the mauls in rugby occur after lineouts. The ball is thrown in to a jumper who then passes the ball back through the maul to the player at the back. If the maul passes over the tryline the ball carrier will dive for the line to score, often causing other players to fall or defenders to dive to try to hold up the ball. During a majority of this process the ball is not visible and the ball carrier can only be identified by observing body positions. Labeling the jumper as ball carrier is often obvious because the ball can be seen above the other players. As the maul is moving forward the rearmost player was labeled as the ball carrier. No ball carrier was labeled if no part of the player is visible in the frame, or if the ball carrier could not be identified from other players on the ground.
##### Tackles
After a player is tackled, other players form a ruck over the ball until the scrum half is able to distribute the ball to other players. This introduces two issues, the tackled player is often occluded by the players forming a ruck, and identifying the moment when the tackled player is no longer in possession of the ball is difficult. The tackled player was labeled as the ball carrier using a best guess for their location until the scrum half could be seen moving to pass the ball.
##### Labeling Imbalance
England dominated the game against Scotland so most of the highlights include England ball carriers and there are fewer frames with Scotland ball carriers. Due to this imbalance the labels were changed to ball_carrier exclusive of the team possessing the ball. 

### Training a Model
#### Custom Train on Roboflow
Roboflow provides options to train models on the platform. I trained a yolov8 model with 70% of the labeled frames in the training set, 20% in the validation set, and 10% in the test set. The model acheived 49.3% mAP, 71.4% precision, and 54% recall. Inspecting the test set showed that most frames with the ball occluded did not have a labeled ball carrier, but it performed better than expected on mauls. It is possible this is because mauls are slow moving and the ball carrier stays in a similar position to the rest of the players for a long time. This could have caused very similar mauling frames to exist in the test set as the train set because the frames were randomized.

#### Yolov7 on Colab Notebook
I wanted to also try training a model on this dataset outside of the Roboflow platform. I found a Roboflow [blog](https://blog.roboflow.com/yolov7-custom-dataset-training-tutorial/) that included a Colab notebook training a yolov7 model on a custom dataset. I used that notebook to learn how to download the labeled frames from Roboflow and train a yolov7 model. The first model was trained for 55 epochs and did not predict ball carrieres in almost any frame. The second model was trained for 300 epochs and was likely to predict multiple ball carriers per frame, with almost all the players labeled as ball carriers being England players even when Scotland had possession. This was likely caused by the dataset having many more frames with English ball carriers than Scottish ball carriers.
#### Predicting Locally
After downloading the weights, I predicted over the highlights (same as training data, just to test ability). On my CPU this took about 14 hrs. The final video had to be exported from the saved .mp4 to .mov in order to play in a jupyter notebook.


Next Steps
- does this need to be accurate to move to the next stage?
    * only if detecting the ball carrier is going to be a critical part of detecting a knock-on
    * or if I want this to be an important part of the project summary
        - if it is just a learning step to find data and train a model then the model doesn't have to be stellar
- add early stopping?
- compare methods Lisa shared
- write a summary

Other Questions
- can I sort the images before uploading to Roboflow? (should make labeling easier if the frames are related instead of skipping all over the video)
