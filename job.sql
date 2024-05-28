DROP TABLE IF EXISTS [job];

-- Create the job table
CREATE TABLE job (
    id INTEGER PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    location VARCHAR(255),
    category VARCHAR(255),
    company_id INT NOT NULL,
    experience VARCHAR(255),
    requirements TEXT,
    status VARCHAR(255),
    salary_range VARCHAR(255), -- Added salary range
    employment_type VARCHAR(255), -- Added employment type (e.g., Full-time, Part-time)
    benefits TEXT, -- Added job benefits
    application_deadline DATE, -- Added application deadline
    FOREIGN KEY (company_id) REFERENCES company(id)
);

-- Dummy data for the job table
INSERT INTO job (title, description, location, category, company_id, experience, requirements, status, salary_range, employment_type, benefits, application_deadline) VALUES
('Software Engineer', 'Developing web applications using Python and Django.', 'New York', 'Software Development', 1, '2+ years', 'Python, Django, HTML, CSS', 'open', '$80,000 - $100,000', 'Full-time', 'Health insurance, 401(k)', '2024-06-30'),
('Data Analyst', 'Analyzing data and creating reports for decision making.', 'San Francisco', 'Data Analysis', 2, '3+ years', 'SQL, Excel, Data Visualization', 'close', '$70,000 - $90,000', 'Full-time', 'Health insurance, Paid time off', '2024-05-31'),
('Marketing Manager', 'Creating and executing marketing campaigns.', 'New York', 'Marketing', 1, '5+ years', 'Marketing strategy, SEO, Social media marketing', 'open', '$90,000 - $120,000', 'Full-time', 'Health insurance, Gym membership', '2024-07-15');
