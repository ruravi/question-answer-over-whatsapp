import unittest
from whatsapp_to_db import WhatsAppToDb


class TestCases(unittest.TestCase):
    def test_parse_txt_to_csv(self):
        whatsapp_to_db = WhatsAppToDb()
        actual_op = whatsapp_to_db._parse_txt_to_csv(
            "data/whatsapp_export_test.txt", "data/intermediate_test.csv"
        )
        self.assertEqual(actual_op.as_posix(), "data/intermediate_test.csv")
        with open(actual_op, "r") as f:
            actual_lines = f.readlines()
            self.assertEqual(actual_lines[0], "date,sender,text\n")
            self.assertEqual(len(actual_lines), 837)

        whatsapp_to_db = WhatsAppToDb()
        whatsapp_to_db._csv_to_db("data/intermediate_test.csv", "data/test.db")
