import os
import unittest
from unittest import mock

from app import api_token


class TokenTests(unittest.TestCase):
    def test_reads_environment(self):
        with mock.patch.dict(os.environ, {"API_TOKEN": "x"}):
            self.assertEqual(api_token(), "x")
