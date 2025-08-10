import mysql.connector
from mysql.connector import Error, pooling
from typing import List, Dict, Any, Optional, Tuple
import os
from contextlib import contextmanager
from utils.logger import SystemLogger
from utils.exceptions import DatabaseConnectionError, DatabaseQueryError


class MySQLConnector:
    """
    MySQL database connector with connection pooling and error handling.
    
    Handles all course catalog operations with proper connection 
    management and transaction support. 
    
    Note: User enrollments are handled by Neo4j graph database.
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 database: str = "course_recommendation", 
                 user: str = "root",
                 password: str = "",
                 pool_name: str = "course_pool",
                 pool_size: int = 5):
        """
        Initialize MySQL connector with connection pooling.
        
        Parameters
        ----------
        host : str, optional
            MySQL server host, by default "localhost"
        database : str, optional 
            Database name, by default "course_recommendation"
        user : str, optional
            MySQL username, by default "root"
        password : str, optional
            MySQL password, by default ""
        pool_name : str, optional
            Connection pool name, by default "course_pool"
        pool_size : int, optional
            Maximum connections in pool, by default 5
        """
        self.config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'autocommit': False,
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
        
        # Create connection pool
        SystemLogger.info(f"Initializing MySQL connection pool", {
            'host': host, 'database': database, 'user': user, 'pool_size': pool_size
        })
        
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name=pool_name,
                pool_size=pool_size,
                pool_reset_session=True,
                **self.config
            )
            SystemLogger.info("MySQL connection pool created successfully", {
                'pool_name': pool_name, 'pool_size': pool_size
            })
        except Error as e:
            SystemLogger.error(
                f"Failed to create MySQL connection pool - Check MySQL server status and credentials",
                exception=e,
                context={'config': {k: v for k, v in self.config.items() if k != 'password'}}
            )
            raise DatabaseConnectionError(f"Failed to create connection pool: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields
        ------
        mysql.connector.connection.MySQLConnection
            Database connection with automatic cleanup
            
        Raises
        ------
        ConnectionError
            If unable to get connection from pool
        """
        connection = None
        SystemLogger.debug("Acquiring MySQL connection from pool")
        
        try:
            connection = self.pool.get_connection()
            SystemLogger.debug("MySQL connection acquired successfully")
            yield connection
        except Error as e:
            SystemLogger.error(
                "Failed to acquire MySQL connection from pool - Pool may be exhausted or MySQL server down",
                exception=e,
                context={'pool_size': getattr(self.pool, '_cnx_queue', {}).maxsize if hasattr(self.pool, '_cnx_queue') else 'unknown'}
            )
            if connection and connection.is_connected():
                connection.rollback()
            raise DatabaseConnectionError(f"Database connection error: {e}")
        finally:
            if connection and connection.is_connected():
                connection.close()
                SystemLogger.debug("MySQL connection returned to pool")
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.
        
        Parameters
        ----------
        query : str
            SQL SELECT query
        params : tuple, optional
            Query parameters for prepared statements
            
        Returns
        -------
        List[Dict[str, Any]]
            Query results as list of dictionaries
            
        Raises
        ------
        Exception
            If query execution fails
        """
        SystemLogger.debug(f"Executing MySQL SELECT query", {'query_preview': query[:100], 'has_params': params is not None})
        
        with self.get_connection() as connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()
                
                SystemLogger.debug(f"MySQL query executed successfully", {
                    'rows_returned': len(results),
                    'query_preview': query[:100]
                })
                return results
            except Error as e:
                SystemLogger.error(
                    f"MySQL SELECT query execution failed - Check query syntax and table existence",
                    exception=e,
                    context={
                        'query': query,
                        'params': params,
                        'error_code': getattr(e, 'errno', 'unknown'),
                        'sql_state': getattr(e, 'sqlstate', 'unknown')
                    }
                )
                raise DatabaseQueryError(f"Query execution failed: {e}")
    
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE query.
        
        Parameters
        ----------
        query : str
            SQL modification query
        params : tuple, optional
            Query parameters for prepared statements
            
        Returns
        -------
        int
            Number of affected rows
            
        Raises
        ------
        Exception
            If query execution fails
        """
        SystemLogger.debug(f"Executing MySQL UPDATE/INSERT/DELETE query", {'query_preview': query[:100], 'has_params': params is not None})
        
        with self.get_connection() as connection:
            try:
                cursor = connection.cursor()
                cursor.execute(query, params)
                connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                
                SystemLogger.debug(f"MySQL update query executed successfully", {
                    'affected_rows': affected_rows,
                    'query_preview': query[:100]
                })
                return affected_rows
            except Error as e:
                connection.rollback()
                SystemLogger.error(
                    f"MySQL UPDATE/INSERT/DELETE query execution failed - Check query syntax and constraints",
                    exception=e,
                    context={
                        'query': query,
                        'params': params,
                        'error_code': getattr(e, 'errno', 'unknown'),
                        'sql_state': getattr(e, 'sqlstate', 'unknown')
                    }
                )
                raise DatabaseQueryError(f"Update query failed: {e}")
    
    def get_courses(self) -> List[Dict[str, Any]]:
        """
        Retrieve all courses with their modules.
        
        Returns
        -------
        List[Dict[str, Any]]
            Course data with modules
        """
        SystemLogger.info("Retrieving all courses from MySQL database")
        
        query = """
        SELECT 
            course_name,
            module_name,
            module_summary
        FROM course_modules_view 
        ORDER BY course_name, order_index, module_name
        """
        
        try:
            courses = self.execute_query(query)
            SystemLogger.info(f"Successfully retrieved courses from database", {
                'total_entries': len(courses),
                'unique_courses': len(set(row['course_name'] for row in courses))
            })
            return courses
        except DatabaseQueryError as e:
            SystemLogger.error(
                "Failed to retrieve courses - Check if course_modules_view exists and is accessible",
                exception=e,
                context={'view_name': 'course_modules_view'}
            )
    
    def search_courses(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search courses and modules by keyword.
        
        Parameters
        ----------
        search_term : str
            Search keyword for course/module names and summaries
            
        Returns
        -------
        List[Dict[str, Any]]
            Matching courses and modules
        """
        query = """
        SELECT 
            course_name,
            module_name, 
            module_summary
        FROM course_modules_view
        WHERE course_name LIKE %s 
           OR module_name LIKE %s 
           OR module_summary LIKE %s
        ORDER BY course_name, order_index, module_name
        """
        search_pattern = f"%{search_term}%"
        return self.execute_query(query, (search_pattern, search_pattern, search_pattern))
    
    def get_course_by_name(self, course_name: str) -> List[Dict[str, Any]]:
        """
        Get specific course with all its modules.
        
        Parameters
        ----------
        course_name : str
            Name of the course to retrieve
            
        Returns
        -------
        List[Dict[str, Any]]
            Course modules data
        """
        query = """
        SELECT 
            course_name,
            module_name,
            module_summary
        FROM course_modules_view 
        WHERE course_name = %s
        ORDER BY order_index, module_name
        """
        return self.execute_query(query, (course_name,))
    
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        SystemLogger.debug("Testing MySQL database connection")
        
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                
            SystemLogger.info("MySQL database connection test successful")
            return True
        except Exception as e:
            SystemLogger.error(
                "MySQL database connection test failed - Check MySQL server status and network connectivity",
                exception=e,
                context={'config': {k: v for k, v in self.config.items() if k != 'password'}}
            )
            return False