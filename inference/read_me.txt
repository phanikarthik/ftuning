1) First generate the QA datasets using the ollama models (this repo). The output of this process is the aplaca_op3.jsonl file
2) Upload the aplaca_op3.jsonl as an input to run train_model.ipynb in Google collab. This will train the model. The trained model
files will be present in ./alpaca-ft. The script will download this folder into local machine.
3) Next upoad this downloaded model to Hugging face.
4) Now we can run the run_mbtn1 script in google collab.