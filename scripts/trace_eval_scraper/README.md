# Web Scraping Automation for TRACE EVAL Data

## Overview
This script automates the process of logging into Northeastern University's TRACE Evaluation system, selecting specific terms and schools, and downloading course evaluation data in Excel format for every available combination of Term, School, Department, Instructor and Course. The script uses Selenium WebDriver for web automation and stores download tracking details in a CSV file.

## Features
- Automated login using credentials stored in environment variables.
- Duo authentication handling.
- Dynamic selection of terms and schools.
- Automated table interactions and data extraction.
- Bulk downloading of course evaluation reports in Excel format.
- Tracking of downloaded files in a CSV report.

## Prerequisites
Ensure you have the following installed:

- Python 3.x
- Google Chrome (133 and above)
- Chrome WebDriver (compatible with your Chrome version)
    - Can be downloaded from here: https://googlechromelabs.github.io/chrome-for-testing/
- Required Python libraries: `selenium`, `pandas`, `python-dotenv`

Install dependencies using:

```sh
pip install selenium pandas python-dotenv
```

## Setup

### 1. Configure Environment Variables
Create a `.env` file in the project directory and add:

```env
NEU_USERNAME=your_username
NEU_PASSWORD=your_password
```

### 2. Update Script Paths
Modify the script to update the following paths:

- `chrome_driver_path`: Path to your Chrome WebDriver.
- `download_directory`: Directory where Excel files will be saved.
- `csv_path`: Path where the download tracking file will be stored.

### 3. Adjust Search Parameters
Update the `terms` and `schools` lists in the script to specify the academic terms and schools you want to filter.

## Usage
Run the script using:

```sh
python script.py
```

The script will:
1. Launch Chrome.
2. Log in using the provided credentials. (2-Factor Authentication through DUO)
3. Navigate to the report browser and apply filters.
4. Click "View All" to load all results.
5. Iterate through each row item, open the evaluation page, and download the Excel report.
6. Save a tracking log of downloaded files.

## Output

- Downloaded Excel reports in the specified directory.
- A CSV file (`download_tracking_<timestamp>.csv`) containing the status of each file.

## Troubleshooting

- **Login Issues**: Ensure your credentials in `.env` are correct.
- **Duo Authentication Timeout**: Approve the Duo push notification promptly.
- **Web Elements Not Found**: Ensure the webpage structure hasn't changed.
- **Slow Loading**: Increase `WebDriverWait` durations if elements are loading slowly.

## License
This project is for academic use only. Unauthorized scraping of websites may violate terms of service.

## Author
Developed by Sanidhya Karnik for the Recommendation System Project at Northeastern University.