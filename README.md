# pos-tagger

- average accuracy of 87%
- uses Viterbi algorithm and Hidden Markov Model 

# Running the program 
Note: you must have python3 to run the program

You have been provided with some training and test files. You can use them to run the program.

use the following command:
  
      python3 ./tagger.py --trainingfiles <training files> --testfile <test file> --outputfile <output file> --answerfile <answer file>
      

for example: 

    python3 ./tagger.py --trainingfiles training1.txt --testfile test2.txt --outputfile new.txt --answerfile training2.txt

* **training files:** the files that will be used to train the HMM, all the words in the file are tagged, pick from the files called training
* **test file:** A file with untagged words that will be tagged, pick from the files called test
* **output file:** the file to which the result is written to, the untagged words from the test file with the predicted tagging, you can name this file anything you like
* **answerfile:** the file with the real tagging for the test file, this argument is optional, it is only used to measure the accuracy of the tagging 
* **Note that the test files and matching answer files have the same number (ex. test2.txt and training2.txt)**
* **the accuracy will be printed out to terminal if the --answerfile argument is passed**

