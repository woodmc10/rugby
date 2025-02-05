Idea - track the improvements of Women's Six Nations rugby teams after players are given professional contracts

Metrics - errors
    - **ball handling errors represented by number of knock-ons per game**
    - missed tackles per game
    - turnovers conceded per game
    - pentalties per game

Project Outline
1. Learn Computer Vision techniques
    - **identify ball carriers in highlights**
    - **identify ball carriers in full games**
    - **identify knock-ons in full games**
    - identify tackles in full games
    - identify missed tackles in full games
    - identify rucks 
    - identify turnovers
    - identify penalties
1. Generate stats for last 10 years of Women's Six Nations games
    - depends on film availability
1. Find details on players' contracts
1. Visualize changes over time


## 2025 Update
- Still working on learning computer vision techniques but focusing on action recognition in sports.

Timeline:
* 1-20: 
    - find an action recognition tutorial
        * https://www.tensorflow.org/tutorials/load_data/video
    - build a model (preferably using transfer learning)

* 1-27:
    - find data (veo videos of 7s if possible), pick a metric (tackles, missed tackles, knockons) to focus on
    - label at least 1 game
        * https://www.youtube.com/watch?v=F2pTiS4yylo

* 2-3: 
    - label more games
    - run a small dataset through an open source model

* 2-10:
    - label more games
    - train transfer learning model

* 2-17:
    - explore other open source models
    - label, train, iterate (and continue exploring other model types)