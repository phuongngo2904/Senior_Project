# Senior Project - Sign Language Translator

# 1. Install pip, python, virtualenv on RaspberryPi OS

### If you already installed python(any versions), please ignore this step:
```
sudo apt-get install python3
```
* Then, verify Installation and check pip and python version:
```
python3 --version
pip3 -V

```
### Install virtual environtment
* Install virtualenv
```
sudo pip3 install virtualenv
```
* Create a directory to store env
```
mkdir environtments
cd environtments
```
* Create a virtual environment
```
virtualenv senior
```
* Activate virtual environment
```
cd senior
source bin/activate
```
* Then go back to the root
```
cd 
```
# 2. Clone the project
* Make sure you have git installed by checking its version:
```
git --version 
```
* Otherwise, install it:

To install Git for Windows, point your browser at https://git-scm.com/download/win. A download of the Windows Git installer will begin automatically. Once complete, you can double-click the installer and follow the steps.<br />
* You can view your default Git configuration options with the following command:
```
git config -h
```
* Clone the project:
```
git clone https://github.com/phuongngo2904/Senior_Project.git
```
* Navigate to this project 
```
cd Senior_Project
```
# 3. Install tensorflow manually
* Check Raspberry Pi (GNU/Linux 10 (Buster)), Python and Pip version.
```
cat /etc/os-release 
python3 --version
pip3 --version 
```
* Make sure everything is updated.
```
sudo apt update
sudo apt dist-upgrade
sudo apt clean
```
* Upgrade pip (>19.0), setuptools (≥41.0.0) and install Tensorflow 2 dependencies.<br />
If you use virtual environtment, in the cmd line 'pip3 install -U --user six wheel mock'<br />
you can ignore the '--user'. You can run this cmd line instead: 'pip3 install -U six wheel mock'<br />
```
sudo pip install --upgrade pip
sudo pip3 install --upgrade setuptools
sudo pip3 install numpy==1.19.0

sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran python-dev libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev

sudo pip3 install keras_applications==1.0.8 --no-deps
sudo pip3 install keras_preprocessing==1.1.0 --no-deps
sudo pip3 install h5py==2.9.0
sudo pip3 install pybind11

pip3 install -U --user six wheel mock
```
* Install Tensorflow 2*
```
wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/main/tensorflow-2.5.0-cp37-none-linux_armv7l_numpy1195_download.sh"

sudo chmod +x tensorflow-2.5.0-cp37-none-linux_armv7l_numpy1195_download.sh

./tensorflow-2.5.0-cp37-none-linux_armv7l_numpy1195_download.sh

sudo pip3 uninstall tensorflow

sudo -H pip3 install tensorflow-2.5.0-cp37-none-linux_armv7l.whl
```
# 4.  Install requirements
```
pip3 install -r requirements.txt
```
# 5. Run the real time recognition
```
cd code
python3 real_time_detection.py
```
# Reference 
* Dataset: https://www.kaggle.com/grassknoted/asl-alphabet
* TFLite model : https://mega.nz/file/WQdAjZZL#IhPKLt4sYcpWtQXMIVQQrDNhF7UvPOF0lv2e5wXAjlc
