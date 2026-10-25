import unittest

import main


class MainTests(unittest.TestCase):
    def test_default_database(self):
        self.assertTrue(main.DATABASE_URL)
