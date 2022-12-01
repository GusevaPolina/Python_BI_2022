# Homework 7: functional programming

These exercises explore the creativity of creating functions in Python. 

 <br />
 
The following functions are created:
- `sequential_map(*args)` applies a list of functions to the last of arguments which is a container;
- `consensus_filter(*args)` applies a list of boolean functions to the last of arguments which is a container;
- `conditional_reduce(func1, func2, container)`: the first function filters a container and the second one acts as `reduce`;
- `func_chain(*args)` combines a list of functions into one superfunction;
- `multiple_partial(*args, **kwargs)` acts as `partial` without importing `partial` by inserting kwargs into a list of functions;
- `printer(*args, **kwargs)` acts as `print` without `flush` using `sys`.

 <br />

 Please find a script and the `requirements.txt` file attached.
