PythonTracer user guide
=======================

.. role:: python(code)
   :language: python

.. role:: console(code)
   :language: console

Install
-------

This guide will help you download PythonTracer step by step.


Step 1 : Download the repository from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to `GitHub project page <https://github.com/abdelneuhaus/PythonTracer>`_.
2. Click on **Code** (green button).
3. Choose **Download ZIP** to download locally the files into your computer.
4. Extract files in an easily accessible directory (for example, :console:`C:\python_tracer`).


Step 2 : Installing Python
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Download Python **3.13** from the `official website <https://www.python.org/downloads/>`_.
2. During the install, be sure to check the **Add Python to PATH** option.
3. Once installed, verify that Python works:

   - Open a powershell or command terminal (:console:`PowerShell` in Windows).
   - Type :console:`python --version` and press **Entrée**

.. note::
   You should see your Python version (:console:`Python 3.x.x`).


Step 3 : Installing CUDA Toolkit (if not already done)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Download CUDA Toolkit **12.6 or above** from the `official website <https://developer.nvidia.com/cuda-12-6-0-download-archive>`_.
2. Launch the install.
3. Once installed, check that CUDA is correclty working:

   - Open a powershell or command terminal (:console:`PowerShell` in Windows).
   - Type :console:`nvcc --version` and press **Entrée**

.. note::
   You should see your CUDA version (:console:`Cuda compilation tools, release 12.x, V12.x.x`).


Step 4 : Creating a virtual environment (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A virtual environment allows to handle project dependencies in an isolated manner.

1. Open a powershell or command terminal (:console:`PowerShell` in Windows) and go to the directory where you have extracted the files (for example, type:console:`cd C:\\palm_tracer`).
2. Then, create the virtual environment by typing in the powershell :console:`python -m venv venv`
3. Activate the virtual environment with :console:`.\venv\Scripts\activate`
4. You should now see :console:`(venv)` before the path in your Powershell, meaning the virtual environment is activated.


Step 5 : Installing PythonTracer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open a powershell or command terminal (:console:`PowerShell` in Windows) and go to the directory where you have extracted the files (for example, type:console:`cd C:\\palm_tracer`).
2. Be sure that the virtual environment is activated if you are using one.
3. Instal required dependencies by typing : :console:`python -m pip install requirements.txt`


Step 6 : Lauching PythonTracer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open a powershell or command terminal (:console:`PowerShell` in Windows) and go to the directory where you have extracted the files (for example, type:console:`cd C:\\palm_tracer`).
2. Be sure that the virtual environment is activated if you are using one.
3. Launch :console:`PythonTracer` by typing : :console:`python main.py`


it's done ! 🎉 You have installed and setup successfully PythonTracer.


FAQ
---

**1. Why using a virtual environment?**
To avoid conflicts between dependencies versions of different projects.

**2. And if `pip install` is not working?**
That means that Python is not correclty installed. Be sure to have checked the `add to PATH` option during the install.
