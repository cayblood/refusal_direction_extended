"""Shared library for the refusal-direction pipeline.

Feature packages (``data``, ``runtime``, ``activations``, ``interventions``,
``transfer``, ``plots``) hold the reusable logic. The numbered entry scripts in
``scripts/`` import from here and stay thin (argparse + a single call).
"""
