# Load Experiment from a Database

Use the Database to initialize the Experiment:

```{.py3 .in py  linenums="1"}
import vodex as vx

# Provide the path to the database file
experiment = vx.Experiment.load("test.db")
```
