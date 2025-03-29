# CSI 4500 Task Manager Optimizer
_______________________________

## Backend
To install all requirements for python, first initiate a venv then install python requirements in terminal. NOTE: In some linux distributions you will use python / pip instead of python3 / pip3
```py
# intializes venv and activates it
python3 -m venv venv
source venv/bin/activate 

# downlaods all requirements.txt files
pip3 install -r REQUIREMENTS.txt
```

Before trying to run the application, please go to your terminal and install the packages located in REQUIREMENTS.txt.

Enter your terminal:

Navigate to the project directory: 

python collect_training_data.py

python train_model.py

Check that ml_model.pkl (and possibly label_encoder.pkl if applicable) exists in your backend folder.

Then launch the app: 

python app.py

## Front End 

### Windows
#### Downlaod fnm (fast node manager): <br>

```
winget install Schniz.fnmwinget install Schniz.fnm
```

### Node.js
Download using Fnm
```
fnm install 22
```
Verify
```
node -v
```
#### If Node is not being recognized

Add the Following to the end of profile file: 
```
fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression
```
If there is no profile, do the following before the previous step
```
if (-not (Test-Path $profile)) { New-Item $profile -Force }
```
Edit profile and put fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression into powershell
```
Invoke-Item $profile
```
https://github.com/Schniz/fnm

https://nodejs.org/en/download

