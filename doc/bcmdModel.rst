bcmdModel
=========

Running the BCMD Model
----------------------
The BCMD model can be run in a number of ways, both using the command lines
and the WeBCMD interface. Over time, both the BayesCMD package and the WeBCMD
package are expected to merge. As a result, the BCMD model class has been
designed to allow flexibility and compatibility with both the current BayesCMD
framework and the WeBCMD framework.

bcmd_model
^^^^^^^^^^
.. automodule:: bayescmd.bcmdModel.bcmd_model

.. autoclass:: bayescmd.bcmdModel.ModelBCMD
    :members:
    :private-members:

Input Creation
--------------
Input files are required by the BCMD model. A special class has been created
that will create a correctly formatted input file for a variety of use cases.

input_creation
^^^^^^^^^^^^^^

.. automodule:: bayescmd.bcmdModel.input_creation

.. autoclass:: bayescmd.bcmdModel.InputCreator
    :members:
