from pathlib import Path
import re, csv
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine


class WhatsAppToDb:
    def __init__(self):
        message_line_regex = r"""
            \[?
            (
                \d{1,2}
                [\/.]
                \d{1,2}
                [\/.]
                \d{2,4}
                ,\s
                \d{1,2}
                :\d{2}
                (?:
                    :\d{2}
                )?
                (?:[ _](?:AM|PM))?
            )
            \]?
            [\s-]*
            ([~\w\s]+)
            [:]+
            \s
            (.+)
        """
        self.regex_pattern = re.compile(message_line_regex, flags=re.VERBOSE)

    def parse_txt_to_db(self, input_file_path: str, output_db_path: str):
        csv_path = self._parse_txt_to_csv(input_file_path, "data/intermediate.csv")
        self._csv_to_db(csv_path, Path(output_db_path))

    def _parse_txt_to_csv(self, input_file_path: str, output_file_path: str) -> Path:
        input_path = Path(input_file_path)
        intermediate_path = Path(output_file_path)

        with open(input_path, encoding="utf8") as input_file, open(
            intermediate_path, "w", encoding="utf8"
        ) as intermediate_file:
            intermediate_file_writer = csv.writer(intermediate_file)
            intermediate_file_writer.writerow(["date", "sender", "text"])
            for line in input_file.readlines():
                result = re.match(self.regex_pattern, line.strip())
                if result:
                    date, sender, text = result.groups()
                    sender = sender.strip("~")
                    sender = sender.strip()
                    datetime_object = datetime.strptime(date, "%m/%d/%y, %I:%M:%S %p")
                    intermediate_file_writer.writerow(
                        [
                            datetime_object.strftime("%Y-%m-%d %H:%M:%S"),
                            sender,
                            text.strip(),
                        ]
                    )

        return intermediate_path

    def _csv_to_db(self, csv_path: Path, output_path: Path):
        # Create a SQLAlchemy engine
        engine = create_engine(f"sqlite:///{output_path}")

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)

        # Write the DataFrame to a SQLite table
        df.to_sql("whatsapp_chat", engine, if_exists="replace", index=False)
