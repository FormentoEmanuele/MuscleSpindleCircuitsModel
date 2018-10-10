# README #

Epidural electrical spinal cord stimulation can improve motor control in animal models and humans with spinal cord injury. The current understanding is that the stimulation excites sensorimotor circuits below the injury by interacting with the natural activity of large afferent fibers entering into the spinal cord from the dorsal roots. Here, we built a computational framework to study the mechanisms of this interaction and how this may differ across species.

This repository contains the code of the neural simulations performed in:  Formento, E. et al., Electrical spinal cord stimulation must preserve proprioception to enable locomotion in humans with spinal cord injury, Nature Neuroscience (2018).

### How do I get set up? ###

* Dependencies
    * python 2.7
        * mpi4py
        * numpy
        * pandas
        * matplotlib
    * openmpi
    * [neuron](http://www.neuron.yale.edu/neuron/download)
        * --with-python
        * --with-mpi

* Configuration

    The folder neuralnetwork/code/mod_files contains different mechanisms that describe the membrane dynamics or particular cell properties necessary for certain Neuron cell models. These files are written in MOD and need to be compiled. For this purpose cd to the code folder with a terminal application and issue the following command:
```
#!shell
    cd neuralnetwork/code/
    nrnivmodl ./mod_files
```
* Running a simulation

    The different simulations described in the associated paper can be executed by running the files in the code/scripts folder.
    Please refer to the comments inside each script to see the required arguments that need to be passed at launch time.
    In general, every script file within this repo need to be run from the code folder.
    For example:

```
#!shell
    cd neuralnetwork/code/
    python scripts/runForSimMuscleSpindles.py 40 240 rat frwSimRat.txt test
```
