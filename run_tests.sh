#! /bin/bash

#Run Unit Test using PyTest ($ pip install pytest)

# Running python tests is path-sensitive.
# Export the PYTHONPATH and change to the directory where this script lives
DIR=$(cd $(dirname "$0"); pwd)
export PYTHONPATH="${PYTHONPATH}:$DIR"
echo $PYTHONPATH
cd $DIR



# Run tests
PYTHONPATH=. py.test tests/
