# DietSmart
Machine learning based app to recognize food from images and provide helpful related information.

## Environment Installation:
I am using a Windows machine and I I found that using Anaconda and the Conda package manager makes it really easy to get started.

1. Ensure you have Anaconda installed, you can download it from https://www.anaconda.com/distribution/#download-section
2. In command line shell create a directory where you want to download the source code.  The folder DietSmart will be created in the next step
3. Git clone this repository using: `git clone https://github.com/sbelal/DietSmart.git`
4. Switch to current repo directory DietSmart
5. To create a Python environment and to install prequisite packaghes run: `conda env create -f environment.yml`
6. [Optional] To update an existing conda environment with the required packages run: `conda env update --file environment.yml`


## Running the app
1. Activate the environment: `conda activate dietsmart`
2. Switch to repo folder DietSmart
3. To run app: `python app.py`
4. Using your fav internet browser (If you use IE or Edge...you have no hope) and go to http://127.0.0.1:5000/
5. Profit :)

