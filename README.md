# Privacy
Privacy preserving synthetic data generation

This project uses the SDV-Framework for synthetic data generation. The process to generate data is assisted with the help of multiple extensions. The generated data can be evaluated by multiple methods to get a better understanding of the quality and the degree of anonymity of the data. Thus, the results are more understandable and trustworthy.

The frontend can be found in the Data-Synthesizer-Angular project.


# Setup
Install the requirements. One file was created by Conda and has the versions numbers (@4.2.4) as an ending. If there are any trouble with those requirements, try the other file. After that, the project can be started by executing:

```
uvicorn fastAPI:app
```

# Extend the project
The project uses nearly everywhere the factory-pattern, which allows an easy way to extend the components. To create a new Evaluator, a new class needs to be created in /evaluators/name.py. The class needs to inherit from BaseEval and made accessible in the factory.py file. The ```_compute()``` method should return a value similar to this: 

```
[{'type': 'quality', 'source': 'className', 'metric': 'MetricInfo', 'name': 'MethodName', 'result': { 'key': 'value'}}] 
```

After this and adding the new evaluation to the frontend, it can be choosen and will be executed.
