"""Unit tests for database."""

import sqlite3
import unittest
import uuid

class TestDatabase(unittest.TestCase):
    """Unit tests for database."""
    def __init__(self, *args, **kwargs) -> None:
        """Initialise units tests and inputs"""
        super(TestDatabase, self).__init__(*args, **kwargs)

    def test_database_strucure(self) -> None:
        """Test that the tables and table attributes are set up correctly."""
        connection = sqlite3.connect(database='school_project/saved_models.db')
        cursor = connection.cursor()

        # Check that 'Models' table is in database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        self.assertIn(member="Models", container=cursor.fetchall()[0])

        # Check that 'Models' table has the correct attributes
        expected_table_info = [(0, 'Model_ID', 'INTEGER', 0, None, 1),
                               (1, 'Dataset', 'TEXT', 1, None, 0),
                               (2, 'File_Location', 'TEXT', 1, None, 0),
                               (3, 'Hidden_Layers_Shape', 'TEXT', 1, None, 0),
                               (4, 'Learning_Rate', 'FLOAT', 1, None, 0),
                               (5, 'Name', 'TEXT', 1, None, 0),
                               (6, 'Train_Dataset_Size', 'INTEGER', 1, None, 0),
                               (7, 'Use_ReLu', 'INTEGER', 1, None, 0)]
        cursor.execute("PRAGMA table_info(Models)")
        table_info = cursor.fetchall()
        for expected_attribute, attribute in zip (expected_table_info, table_info):
            for expected_info, info in zip(expected_attribute, attribute):
                self.assertEqual(first=expected_info, second=info)

    def test_not_null_constraint(self) -> None:
        """Test that a the NOT NULL constraint is setup."""
        connection = sqlite3.connect(database='school_project/saved_models.db')
        cursor = connection.cursor()

        # Try to insert with the last attribute missing
        test_data = ("Test_Dataset", 
                     f"school_project/saved-models/{uuid.uuid4().hex}.npz",
                     "100, 100",
                     0.1,
                     "Test_Name",
                     100)
        sql = """
        INSERT INTO Models
        (Dataset, File_Location, Hidden_Layers_Shape, Learning_Rate, Name, Train_Dataset_Size)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        self.assertRaises(sqlite3.IntegrityError, cursor.execute, sql, test_data)

    def test_unique_constraint(self) -> None:
        """Test that the UNIQUE (Dataset, Name) constraint is setup."""
        connection = sqlite3.connect(database='school_project/saved_models.db')
        cursor = connection.cursor()

        # Save test data
        test_data = ("Test_Dataset", 
                     f"school_project/saved-models/{uuid.uuid4().hex}.npz",
                     "100, 100",
                     0.1,
                     "Test_Name",
                     100,
                     True)
        sql = """
        INSERT INTO Models
        (Dataset, File_Location, Hidden_Layers_Shape, Learning_Rate, Name, Train_Dataset_Size, Use_ReLu)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, test_data)
        connection.commit()

        # Try to save the same data again
        test_data = ("Test_Dataset", 
                     f"school_project/saved-models/{uuid.uuid4().hex}.npz",
                     "100, 100",
                     0.1,
                     "Test_Name",
                     100,
                     True)
        sql = """
        INSERT INTO Models
        (Dataset, File_Location, Hidden_Layers_Shape, Learning_Rate, Name, Train_Dataset_Size, Use_ReLu)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.assertRaises(sqlite3.IntegrityError, cursor.execute, sql, test_data)

        # Remove test data from database
        sql = """
        DELETE FROM Models WHERE Dataset=? AND Name=?
        """
        parameters = (test_data[0], test_data[4])
        cursor.execute(sql, parameters)
        connection.commit()

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
