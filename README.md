# Information Retrieval and Data Mining Coursework 2017
### Learning to Rank
##### Group 20 - Anna Aleksieva, Hugo Chu, Raluca Georgescu, Julija Bainiaksinaite

This project evaluates a range of Learning to Rank algorithms RankNet, LambdaMART and AdaRank, together with an implementation of ranking as a classification task and proposes an improved approach based on neural networks.

The following are instructions to run each algorithm proposed:

- RankNet

- LambdaMART
  - The implementation of LambdaMART has made use of the RankPy Library (https://bitbucket.org/tunystom/rankpy)
  - Assumptions:
    - The RankPy library is installed in the LambdaMART directory by following the instructions in the link above.
    - The MSLR-WEB10K data folder is existing in the same directory.
  - First, run the prep.py script to convert the data .txt files to binary for easier loading.
  ```
  python prep.py
  ```
  - All the pretrained models are placed in the subdirectory models/. To reload the pretrained models and evaluate them on nDCG@10 and MAP, run the following command within the directory:
  ```
  python reload.py
  ```
  - To run the grid search on hyperparameters:
  ```
  python grid_models.py
  ```
  - To train a model for each fold and average over the results for nDCG@10 and MAP:
  ```
  python model_folds.py
  ```

- AdaRank
  - Based on the implementation from RankLib, Lemur Project, the code has been run using the following rules from the command line. The user has to be in the same directory with the downloaded RankLib-2.1-patched.jar library file. And has to have access to the MSLR-WEB10K dataset, also from the same current folder.
  ```
  java -jar RankLib-2.1-patched.jar -train MSLR-WEB10K/Fold1/train.txt -test MSLR-WEB10K/Fold1/test.txt -validate MSLR-WEB10K/Fold1/vali.txt -ranker 6 -metric2t NDCG@10 -metric2T ERR@10 -save model_fold1_NDCG.txt -noeq -max 1
  ```
    ```
  java -jar RankLib-2.1-patched.jar -train MSLR-WEB10K/Fold1/train.txt -test MSLR-WEB10K/Fold1/test.txt -validate MSLR-WEB10K/Fold1/vali.txt -ranker 6 -metric2t MAP -metric2T ERR@10 -save model_fold1_MAP.txt -noeq -max 1
  ```
  - Examples refer only to Fold1, but they should be run for all 5 folds and then results would be averaged.
  - In order to obtain the ranking scores, the following command was run (again, command is repeated for all 5 folds and results are averaged):
  ```
  java -jar RankLib-2.1-patched.jar -rank MSLR-WEB10K/Fold1/test.txt -load model_fold1_NDCG.txt -score scores_NDCG.txt
  ```
  - The code for evaluating the rankings against the implemented NDCG@10 and MAP metrics is available in the [AdaRank/AdaRank_Evaluation](https://github.com/RalucaGeorgescu/IRDM/blob/master/AdaRank/AdaRank_Evaluation.ipynb) notebook.

- Ranking as Classification

- Neural Network
