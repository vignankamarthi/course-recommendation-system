#!/usr/bin/env python3
"""
Data Migration Script: CSV to MySQL
==================================

Migrates course and module data from CSV files to MySQL database.
Run this script after setting up MySQL server and creating the database.

Usage:
    python scripts/migrate_to_mysql.py

Requirements:
    - MySQL server running
    - Database 'course_recommendation' created
    - MySQL credentials configured
"""

import sys
import os
import pandas as pd
import mysql.connector
from mysql.connector import Error

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.mysql_connector import MySQLConnector


class DataMigrator:
    """Handles migration of CSV data to MySQL database."""
    
    def __init__(self, mysql_config: dict):
        """
        Initialize migrator with MySQL configuration.
        
        Parameters
        ----------
        mysql_config : dict
            MySQL connection configuration
        """
        self.mysql_config = mysql_config
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "data")
        
    def setup_database(self) -> None:
        """
        Create database and tables using schema file.
        
        Raises
        ------
        Exception
            If database setup fails
        """
        print("Setting up database schema...")
        
        # Read schema file
        schema_path = os.path.join(self.project_root, "src", "database", "schema.sql")
        
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Split into individual statements
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        # Execute each statement
        try:
            connection = mysql.connector.connect(**self.mysql_config)
            cursor = connection.cursor()
            
            for statement in statements:
                if statement:  # Skip empty statements
                    print(f"Executing: {statement[:50]}...")
                    cursor.execute(statement)
            
            connection.commit()
            cursor.close()
            connection.close()
            print("✓ Database schema created successfully")
            
        except Error as e:
            raise Exception(f"Database setup failed: {e}")
    
    def migrate_courses_and_modules(self) -> None:
        """
        Migrate course and module data from CSV.
        
        Raises
        ------
        Exception
            If migration fails
        """
        print("Migrating courses and modules...")
        
        # Read CSV data
        csv_path = os.path.join(self.data_dir, "Course_Module_New.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Course data CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} course-module pairs")
        
        try:
            connector = MySQLConnector(**self.mysql_config)
            
            # Track course IDs for modules
            course_ids = {}
            
            # Insert courses first
            unique_courses = df['Courses'].unique()
            for course_name in unique_courses:
                query = "INSERT IGNORE INTO courses (name) VALUES (%s)"
                connector.execute_update(query, (course_name,))
            
            # Get course IDs
            courses = connector.execute_query("SELECT id, name FROM courses")
            for course in courses:
                course_ids[course['name']] = course['id']
            
            print(f"✓ Inserted {len(unique_courses)} courses")
            
            # Insert modules
            module_count = 0
            for _, row in df.iterrows():
                course_id = course_ids[row['Courses']]
                query = """
                INSERT IGNORE INTO modules (course_id, name, summary, order_index) 
                VALUES (%s, %s, %s, %s)
                """
                params = (course_id, row['Modules'], row['Summary'], module_count)
                connector.execute_update(query, params)
                module_count += 1
            
            print(f"✓ Inserted {len(df)} modules")
            
        except Exception as e:
            raise Exception(f"Course migration failed: {e}")
    
    def note_enrollment_handling(self) -> None:
        """
        Display note about enrollment data handling.
        
        Note: Enrollment data is handled by Neo4j graph database,
        not MySQL. This maintains separation of concerns where
        MySQL stores static course catalog and Neo4j handles
        dynamic user relationships and enrollments.
        """
        print("Note: Enrollment data handled by Neo4j graph database")
        print("  - MySQL: Course catalog (static data)")
        print("  - Neo4j: User relationships and enrollments (dynamic data)")
        
        # Check if enrollment CSV exists for reference
        csv_path = os.path.join(self.data_dir, "enrollment_data_updated.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"  - Found {len(df)} enrollment records in CSV (for Neo4j migration)")
        else:
            print("  - No enrollment CSV found")
    
    def verify_migration(self) -> None:
        """
        Verify the migration was successful.
        
        Raises
        ------
        Exception
            If verification fails
        """
        print("Verifying migration...")
        
        try:
            connector = MySQLConnector(**self.mysql_config)
            
            # Check courses
            courses = connector.execute_query("SELECT COUNT(*) as count FROM courses")
            course_count = courses[0]['count']
            
            # Check modules  
            modules = connector.execute_query("SELECT COUNT(*) as count FROM modules")
            module_count = modules[0]['count']
            
            print(f"✓ Migration verified:")
            print(f"  - Courses: {course_count}")
            print(f"  - Modules: {module_count}")
            print(f"  - Enrollments: handled by Neo4j (not MySQL)")
            
            # Test a sample query
            sample = connector.execute_query(
                "SELECT course_name, module_name FROM course_modules_view LIMIT 3"
            )
            print(f"✓ Sample data accessible: {len(sample)} records")
            
        except Exception as e:
            raise Exception(f"Migration verification failed: {e}")


def main():
    """Main migration function."""
    print("=== Course Recommendation System: CSV to MySQL Migration ===\n")
    
    # MySQL configuration - Update these values!
    mysql_config = {
        'host': 'localhost',
        'database': 'course_recommendation',
        'user': 'root',
        'password': '',  # Update with your MySQL password
    }
    
    print("MySQL Configuration:")
    for key, value in mysql_config.items():
        display_value = '***' if key == 'password' and value else value
        print(f"  {key}: {display_value}")
    print()
    
    try:
        migrator = DataMigrator(mysql_config)
        
        # Run migration steps
        migrator.setup_database()
        migrator.migrate_courses_and_modules()
        migrator.note_enrollment_handling()
        migrator.verify_migration()
        
        print("\n🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("1. Update DatabaseAgent to use MySQL instead of CSV")
        print("2. Ensure Neo4j contains enrollment data") 
        print("3. Test the application end-to-end")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure MySQL server is running")
        print("2. Create database: CREATE DATABASE course_recommendation;")
        print("3. Update MySQL credentials in this script")
        sys.exit(1)


if __name__ == "__main__":
    main()