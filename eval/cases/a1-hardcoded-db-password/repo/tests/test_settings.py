import unittest

import settings


class SettingsTests(unittest.TestCase):
    def test_database_name(self):
        self.assertEqual(settings.DB_NAME, "reports")
