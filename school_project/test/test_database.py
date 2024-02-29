"""Unit tests for database."""

import sqlite3
import unittest
import uuid

class TestDatabase(unittest.TestCase):
    """Unit tests for database."""
    def __init__(self, *args, **kwargs) -> None:
        """Initialise units tests and inputs"""
        super(TestDatabase, self).__init__(*args, **kwargs)

    def test_save_retrieve_consistency(self) -> None:
        """Test that data is not changed between saving and retrieving."""
        connection = sqlite3.connect(database='school_project/saved_models.db')
        cursor = connection.cursor()
        
        test_data = ("Test_Dataset", 
                     f"school_project/saved-models/{uuid.uuid4().hex}.npz",
                     "100, 100",
                     0.1,
                     "Test_Name",
                     100,
                     True)
        
        # Save test data
        sql = """
        INSERT INTO Models
        (Dataset, File_Location, Hidden_Layers_Shape, Learning_Rate, Name, Train_Dataset_Size, Use_ReLu)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, test_data)
        connection.commit()

        # Retrieve test data
        sql = """
        SELECT * FROM Models WHERE Dataset=? AND Name=?
        """
        cursor.execute(sql, (test_data[0], test_data[4]))
        retrieved_data = cursor.fetchall()[0]

        # Remove test data from database
        sql = """
        DELETE FROM Models WHERE Dataset=? AND Name=?
        """
        parameters = (test_data[0], test_data[4])
        cursor.execute(sql, parameters)
        connection.commit()

        # Compare initial test data with retrieved data
        for test_value, retrieved_value in zip(test_data, retrieved_data[1:]):
            self.assertEqual(first=test_value, second=retrieved_value)
