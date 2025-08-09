#TODO: This bad larry is useless legacy technical debt 

#%%
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global constants for paths and credentials
chrome_driver_path = r"C:\Users\karni\OneDrive\Documents\chromedriver-win64\chromedriver.exe"
download_directory = r"C:\Users\karni\OneDrive\Documents\NEU - Spring'25\Recommendation system project\TRACE_EVAL_data\Script - Testing"
csv_path = r"C:\Users\karni\OneDrive\Documents\NEU - Spring'25\Recommendation system project\TRACE_EVAL_data\Script - Testing\download_tracking.csv"
neu_username = os.getenv("NEU_USERNAME")
neu_password = os.getenv("NEU_PASSWORD")
terms = ['Summer 2 2024']
schools = ["Coll of Computer Info Sci"]

# Initialize WebDriver
def initialize_driver() -> webdriver.Chrome:
    service = Service(chrome_driver_path)
    chrome_options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_directory}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Log in to the website
def login(driver: webdriver.Chrome) -> None:
    driver.get("https://www.applyweb.com/eval/shibboleth/neu/36892")
    wait = WebDriverWait(driver, 10)

    # Log in credentials
    username_field = wait.until(EC.presence_of_element_located((By.ID, "username")))
    username_field.send_keys(neu_username)
    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(neu_password)

    login_button = driver.find_element(By.NAME, "_eventId_proceed")
    login_button.click()

    try:
        duo_iframe = wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
        driver.switch_to.frame(duo_iframe)
        push_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Send Me a Push')]")))
        push_button.click()
        print("Duo Push Sent! Approve it on your phone.")
        driver.switch_to.default_content()
        print("Login Successful!")
    except Exception as e:
        print(f"Error handling Duo Push: {e}")
    time.sleep(5)  # Time delay to authenticate on the app manually

# Select term and school filters
def select_filters(driver: webdriver.Chrome, terms: List[str], schools: List[str]) -> None:
    wait_5 = WebDriverWait(driver, 5)

    report_browser_url = "https://www.applyweb.com/eval/new/reportbrowser"
    driver.get(report_browser_url)
    time.sleep(3)
    print("Successfully navigated to Report Browser!")

    filter_iframe = wait_5.until(EC.presence_of_element_located((By.ID, "contentFrame")))
    driver.switch_to.frame(filter_iframe)

    # Selecting terms dynamically
    term_select = wait_5.until(EC.presence_of_element_located((By.ID, "TermSelect")))
    term_list = Select(term_select)
    for term in terms:
        term_list.select_by_visible_text(term)
        print(f"Selected Term as '{term}'")
    time.sleep(2)

    # Selecting schools dynamically
    school_select = wait_5.until(EC.presence_of_element_located((By.ID, "SchoolSelect")))
    school_list = Select(school_select)
    for school in schools:
        school_list.select_by_visible_text(school)
        print(f"Selected School as '{school}'")

# Click view all button and confirm
def view_all_data(driver: webdriver.Chrome) -> None:
    wait = WebDriverWait(driver, 10)
    try:
        view_all_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@ng-click, 'viewAll()')]")))
        view_all_button.click()
        print("Clicked 'View All' successfully!")

        yes_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@ng-click='confirm()']")))
        yes_button.click()
        print("Clicked 'Yes' successfully!")
        time.sleep(30)  # Time delay to let all the rows load in a single page
    except Exception as e:
        print(f"Error during 'View All' or 'Yes' click: {e}")

# Download course data
def download_course_data(driver: webdriver.Chrome) -> List[List[str]]:
    wait_30 = WebDriverWait(driver, 30)
    course_data = []
    table = wait_30.until(EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'table')]")))
    rows = table.find_elements(By.XPATH, ".//tr[contains(@ng-repeat, 'course in evaluatedCourses')]")

    for index, row in enumerate(rows):
        try:
            filter_iframe = wait_30.until(EC.presence_of_element_located((By.ID, "contentFrame")))
            driver.switch_to.frame(filter_iframe)
        except:
            pass
        try:
            print(f"Processing row {index + 1}...")

            term = row.find_element(By.XPATH, "./td[3]").text
            course_code = row.find_element(By.XPATH, "./td[4]").text
            course_name = row.find_element(By.XPATH, "./td[5]").text
            course_type = row.find_element(By.XPATH, "./td[6]").text
            instructor = row.find_element(By.XPATH, "./td[7]").text

            view_link = row.find_element(By.XPATH, ".//td/a[contains(@href, 'coursereport')]").get_attribute("href")
            driver.execute_script(f"window.open('{view_link}');")
            driver.switch_to.window(driver.window_handles[-1])

            export_iframe = wait_30.until(EC.presence_of_element_located((By.ID, "contentFrame")))
            driver.switch_to.frame(export_iframe)

            export_link = wait_30.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@title, 'Export As Excel')]")))
            export_link.click()
            print(f"Exported Excel for {course_code} - {course_name} successfully!")

            course_data.append([term, course_code, course_name, course_type, instructor, "Downloaded"])
            driver.close()
            time.sleep(2)
            driver.switch_to.window(driver.window_handles[0])
            time.sleep(2)

        except Exception as e:
            print(f"Error processing row {index + 1}: {e}")
            course_data.append([term, course_code, course_name, course_type, instructor, "Failed"])

    return course_data

# Save course data to CSV
def save_data_to_csv(course_data: List[List[str]]) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"download_tracking_{timestamp}.csv"
    csv_filepath = os.path.join(download_directory, csv_filename)
    
    course_ratings = pd.DataFrame(course_data, columns=["Term", "Course Code", "Course Name", "Type", "Instructor", "Status"])
    course_ratings.to_csv(csv_filepath, index=False)
    print(f"Tracking file saved at: {csv_filepath}")

# Main script execution
def main() -> None:
    driver = initialize_driver()
    login(driver)
    select_filters(driver, terms, schools)
    view_all_data(driver)
    course_data = download_course_data(driver)
    save_data_to_csv(course_data)
    print("Script execution completed!")
    driver.quit()

if __name__ == "__main__":
    main()