import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["tf"] = tf
