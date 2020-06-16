from datetime import datetime
import json


class ResultSaver:

    def save_results_to_db(self, file_id: int, filename: str, store_path: str) -> bool:
        """
        Dumps processing results into the database
        :param file_id:
        :param filename:
        :param store_path:
        :return:
        """
        try:
            # Creates a collection on the fly or finds the existing one
            prediction_results = self.database.db.predictions
            prediction_results.insert(
                {
                    "request_id": self.progress[file_id]["request_id"],
                    "file_id": file_id,
                    "file_name": filename,
                    "saved_to": store_path,
                    "datetime": datetime.utcnow(),
                    "defects": self.progress[file_id]["defects"]
                }
            )
            return True
        except Exception as e:
            print(f"Error while inserting into db: {e}")
            return False