-- User table
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(128) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    name VARCHAR(128) NOT NULL,
    address VARCHAR(255) NOT NULL,
    gender VARCHAR(16) NOT NULL,
    age INT NOT NULL,
    profession VARCHAR(128) NOT NULL,
    -- New fields for API key and usage limit
    gemini_api_key TEXT NULL,
    extraction_uses INT DEFAULT 0,
    extraction_uses_date DATE NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PDF Uploads
CREATE TABLE IF NOT EXISTS uploads (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    filename VARCHAR(255),
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Extraction Jobs
CREATE TABLE IF NOT EXISTS extraction_jobs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    upload_id INT,
    status VARCHAR(32),
    result_file VARCHAR(255),
    pid INT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP NULL,
    error_message TEXT,
    FOREIGN KEY (upload_id) REFERENCES uploads(id)
);

-- Dedupe Results (persisted saved files)
CREATE TABLE IF NOT EXISTS dedupe_results (
    id INT PRIMARY KEY AUTO_INCREMENT,
    job_id INT,
    fields_compared VARCHAR(255),
    result_file VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES extraction_jobs(id)
);

-- Dedupe Runs (audit/log of each run, including temporary files)
CREATE TABLE IF NOT EXISTS dedupe_runs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    job_id INT NOT NULL,
    fields_compared VARCHAR(255) NOT NULL,
    result_file VARCHAR(255) NULL,
    status VARCHAR(32) NOT NULL, -- temp | saved | discarded
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (job_id) REFERENCES extraction_jobs(id)
);
