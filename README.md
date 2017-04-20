# Information Retrieval and Data Mining Coursework 2017
### Learning to Rank
##### Group 20 - Anna Aleksieva, Hugo Chu, Raluca Georgescu, Julija Bainiaksinaite

This project evaluates a range of Learning to Rank algorithms RankNet, LambdaMART and AdaRank, together with an implementation of ranking as a classification task and proposes an improved approach based on neural networks.

The following are instructions to run each algorithm proposed:

- RankNet
  - RankNet algorithm was implemented using open source Lemur Project software, RankLib library (version used: RankLib-2.8.jar). The software has been run using the below commands directly from the command line. User should be in the same directory as RankLib-2.8.jar file, have access to MSLR-WEB10K dataset and follow the following commands: 
  - To train the RankNet model with 5 fold cross validation using the train.txt files from Fold1 (evaluation metrics NDCG@10):
  ```
  java -jar RankLib-2.8.jar -kcv 5 -train MSLR-WEB10K/Fold1/train.txt -ranker 1 -epoch 50 -layer 2 -node 35 -lr 0.0001 -metric2T NDCG@10  -save mymodelranknet14_trial11.txt -kcvmd ~/Desktop -kcvmn mymodelranknet14_trial11_crval.txt -tvs 0.7
  ```
  - To train the RankNet model using the train.txt, val.txt and test.txt files (without cross validation) from Fold1 (evaluation metrics NDCG@10):
  ```
  java -jar RankLib-2.8.jar -train MSLR-WEB10K/Fold1/train.txt -validate MSLR-WEB10K/Fold1/vali.txt -test MSLR-WEB10K/Fold1/test.txt -ranker 1  -epoch 50 -layer 2 -node 100 -lr 0.00005 -metric2t NDCG@10 -save IRDMCW/modelranknet19_trial14.txt
  ```
  - The above examples refers to the data in Fold1 only, due to computational power restrictions, the models have been run separetly on all 5 folds (Fold1-Fold5) and then the results averaged. The models have been trained using NDCG@10 evaluation metrics. 
  - After the models have been trained and parameters tuned, they have been used to rerank the test data and provide with the new ranking for test.txt files, the following command has been used to do reranking:
  ```
  java -jar RankLib-2.8.jar -load IRDMCW/modelranknet19_trial14.txt -rank MSLR-WEB10K/Fold1/test.txt -score IRDMCW2017/reranking_trial14.txt
  ```
  - In order to evaluate the performance of the models, new reranked files were tested agains the NDCG@10, MAP evaluation metrics together with hypothesis testing Student's t-test metrics. The code implemented for the evaluation can be found [here].
 
 
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
