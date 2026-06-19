Installation
============

Requirements
------------

GEMA requires Python 3.9 or later. All runtime dependencies are listed in
``requirements.txt`` and are installed automatically when you install the
package via pip.

Install from PyPI
-----------------

.. code-block:: bash

   pip install GEMA

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/ufvceiec/GEMA.git
   cd GEMA
   pip install -e .

Running the tests
-----------------

.. code-block:: bash

   pip install -e ".[test]"
   python -m pytest tests/ -v
