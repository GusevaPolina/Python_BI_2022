# The third homework

The work was performed on macOS Monterey 12.6 with the Intel Core i5 processor and on python v3.11.Orc2.
  > if you do not possess this version of python, go to the [official website](https://www.python.org/downloads/macos/) and download the mentioned one. After that, install it following the instructions, open the folder with Python3.11 in Applications and run the *Install Certificates.command* file. Now, u r a proud owner of working python v3.11!


Now it's time to shine:
0. open **Terminal** without terminating urself, we have ultraviolence ahead

1.   create a virtual environment with python v3.11

```
python3.11 -m venv <environment_name>
```

2.   activate it 

```
source <environment_name>/bin/activate
```

3.   go to it and download all side packages (move a file with requirements to the folder or create it via vim/another programme)

```
cd <environment_name>
pip install -r requirements.txt  
```

4.   in the name of debugging, go to the guts of the environment and change a small detail ;)

```
cd lib/python3.11/site-packages/pandas/core/
```
Return to the working directory as u returning to drinking in times of despair

```
cd -
```
  > another way to do that is to use pandas==1.4.4 (that's the latest from the previous generation, but 1.4.3 is working or earlier too) 

5. open the *frame.py* file and mute/delete/silence in any of ways both 640 and 641 rows (with vim/another favourite programme)

6. run the *ultraviolence.py* file and enjoy the output! Cos u r gorgeous :sparkles: 

> if it is not in the working directory then move it
```
mv <path_from>/ultraviolence.py <path_to>/<environment_name>/ultraviolence.py
```
> if u do not have it, then create by copypasting! ur university degree should not be wasted

7. do not forget to deactivate ur magnificent virtual environment!

```
deactivate
```

8. also, pls stay hydrated and take it easy, this world needs no more ultraviolence...
