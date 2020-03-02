import pytest
import numpy as np

@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
	doctest_namespace["np"] = np