Python API Reference
====================

GPUCov can be used as a library in addition to the CLI.


``gpucov.instrumenter``
-----------------------

.. automodule:: gpucov.instrumenter
   :members: find_executable_lines, instrument_file, instrument_files, CounterMapping, InstrumentationResult
   :undoc-members:


``gpucov.collector``
--------------------

.. automodule:: gpucov.collector
   :members: read_counter_dump, read_mapping, collect_coverage, generate_lcov, generate_summary, collect_and_report, LineCoverage
   :undoc-members:
