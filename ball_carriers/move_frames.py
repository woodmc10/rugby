import os
import shutil

# split frames between training and extra to limit the amount of labeling necessary
source_folder = r'/Users/marthawood/Documents/repos/rugby/data/23_Scotland_Wales_highlights/'
train_folder = r'/Users/marthawood/Documents/repos/rugby/data/23_Scotland_Wales_highlights/training_frames/'
extra_folder = r'/Users/marthawood/Documents/repos/rugby/data/23_Scotland_Wales_highlights/extra_frames/'
holdout_folder = r'/Users/marthawood/Documents/repos/rugby/data/23_Scotland_Wales_highlights/holdout_frames/'

# for file_name in os.listdir(source_folder):
#     # construct full file path
#     if file_name[-3:] != 'jpg': continue
#     source = source_folder + file_name
#     file_number = int(file_name.replace('.jpg', ''))
#     if file_number % 13 == 0:
#         destination = train_folder + file_name
#     else:
#         destination = extra_folder + file_name
#     # move only files
#     if os.path.isfile(source):
#         shutil.move(source, destination)
#         print('Moved:', file_name)


# move hand-selected frames for holdout testing - selected last try by each team for holdout
scotland_initial_frame = 10972
scotland_final_frame = 11232
wales_initial_frame = 12701
wales_final_frame = 13065

for file_name in os.listdir(train_folder):
    # construct full file path
    if file_name[-3:] != 'jpg': continue
    source = train_folder + file_name
    file_number = int(file_name.replace('.jpg', ''))
    if scotland_initial_frame <= file_number <= scotland_final_frame:
        destination = holdout_folder + file_name
    elif wales_initial_frame <= file_number <= wales_final_frame:
        destination = holdout_folder + file_name
    else: continue
    # move only files
    if os.path.isfile(source):
        shutil.move(source, destination)
        print('Moved:', file_name)