SET "MYPYTHONPATH=C:\Users\Martin\anaconda3"

rem Make sure that PATH is as simple as possible... For Anaconda notice "\Libary\bin"
set PATH=C:\MinGW\mingw64\bin;C:\MinGW\mingw64\lib;%MYPYTHONPATH%;%MYPYTHONPATH%\Scripts;%MYPYTHONPATH%\Library\bin

pip uninstall -y rocketcea
pip cache remove rocketcea
pip install --global-option build_ext --global-option --compiler=mingw32 rocketcea

rem Test the compiled module
python -c "from rocketcea.cea_obj import CEA_Obj; C=CEA_Obj(oxName='LOX', fuelName='LH2'); print(C.get_Isp())"

pause