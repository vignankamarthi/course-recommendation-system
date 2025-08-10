-- Course Recommendation System Database Schema
-- Designed for MySQL 8.0+
-- 
-- PURPOSE: Course catalog storage (static data)
-- NOTE: User enrollments handled by Neo4j graph database

-- Drop tables if they exist (for fresh setup)
DROP TABLE IF EXISTS modules;
DROP TABLE IF EXISTS courses;

-- Courses table: Stores unique course information
CREATE TABLE courses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_course_name (name)  -- Speed up course search queries
);

-- Modules table: Stores course modules with detailed information
CREATE TABLE modules (
    id INT AUTO_INCREMENT PRIMARY KEY,
    course_id INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    summary TEXT NOT NULL,
    order_index INT DEFAULT 0,  -- Module ordering within course
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
    INDEX idx_module_course (course_id),  -- Speed up "get modules for course" queries
    INDEX idx_module_name (name),         -- Speed up module search queries
    UNIQUE KEY unique_module_per_course (course_id, name)  -- Prevent duplicate modules
);

-- Create view for easy course+module queries
CREATE VIEW course_modules_view AS
SELECT 
    c.id as course_id,
    c.name as course_name,
    m.id as module_id,
    m.name as module_name,
    m.summary as module_summary,
    m.order_index
FROM courses c
JOIN modules m ON c.id = m.course_id
ORDER BY c.name, m.order_index, m.name;