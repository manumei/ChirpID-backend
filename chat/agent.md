Revise my entire code. See how ive built the code base for a model that processes bird audios to predict the species based on the recorded sounds.

My current processing pipeline for the training process is:

**Data Fractioning**
1. Get a fraction of the dataset, keeping only birds in a certain region around Argentina
2. Keep only samples with high rating
3. Keep only species where I have several samples and segments
4. Keep only species where I have enough unique authors
5. Load all the audios that meet these conditions
6. Split them into dev/test, grouping by author

**Audio Processing (only for Dev set)**
7. Load the audios with librosa
8. Cut into segments of 5 seconds, only keep the segments with high enough RMS energy, add them as new samples
9. From all the 5-second samples, create grayscale mel-spectrograms

**Data Loading (only for Dev set)**
10. Load the spectrograms and get their grayscale pixels in matrix shape
11. Split into Train/Val grouping by author
12. Load as PyTorch tensors
13. Feed to the PyTorch CNN for training

In my tasks/notebooks.md as I attached in context, I describe what each of the Jupyter Notebooks is meant to do. Most of the first are finished and should not be changed directly, however, for this prompt I do want you to edit the DevTraining.ipynb, ModelTraining.ipynb and ModelSweeping.ipynb notebooks. I want you to revise the utils in split.py, util.py and models.py, and I want you to clean up the entire code. I of course do not want you to delete any models or any of the functions we do use, but the code in the last notebooks and newer functions in util and split has gotten very disorganized and is quite a mess to understand.

We may need to create more structured auxiliary files instead of just lumping everything into util.py and a couple of things onto split. Please organize the auxiliary functions based on their usage, into different auxiliary files inside utils/, and propagate this refactorization throughout all the notebooks that use it, while also maybe propagating refactorization from one utils/ file to another, so the code is very clean and structured into separate files. Enforce modularity, clarify boundaries, remove redundancy.

Make sure you check out tasks/notebooks.md to understand what the notebooks are supposed to do (keeping in mind that some of course aren't finished yet), and after this clean-up, please create an equivalent tasks/utils.md that describes what each of the files in utils/ does. Also create a report of all the changes you have executed, writing it in a new file at tasks/agent_report_1.md
