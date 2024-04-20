import os
import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

# Twitter credentials
username = "Kairos1024"
password = "kairos1921"

def scrape_twitter_info(handles):
    for handle in handles:
        try:
            # Open Twitter handle page
            target_url = f"https://twitter.com/{handle}"
            driver = webdriver.Chrome()
            driver.get(target_url)
            time.sleep(3)
            resp = driver.page_source
            driver.close()

            soup = BeautifulSoup(resp, 'html.parser')

            data = {}
            data["Handle"] = handle

            try:
                data["Account Name"] = soup.find("div", {"class": "r-1vr29t4"}).text
            except:
                data["Account Name"] = None

            profile_header = soup.find("div", {"data-testid": "UserProfileHeader_Items"})

            try:
                data["Followers"] = soup.find_all("a", {"class": "r-rjixqe"})[1].text
            except:
                data["Followers"] = None

            try:
                data["Following"] = soup.find_all("a", {"class": "r-rjixqe"})[0].text
            except:
                data["Following"] = None

            try:
                data["Profile Link"] = target_url
            except:
                data["Profile Link"] = None

            try:
                data["Joined Date"] = profile_header.find("span", {"data-testid": "UserJoinDate"}).text
            except:
                data["Joined Date"] = None

            try:
                data["Post Count"] = soup.find("div", {"class": "r-3s2u2q"}).text
            except:
                data["Post Count"] = None

            # Writing to CSV file
            with open("twitter_scrapping.csv", "a", newline="", encoding="utf-8") as csvfile:
                fieldnames = list(data.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(data)
        except Exception as e:
            print(f"Error scraping handle '{handle}': {e}")

# Initialize Chrome WebDriver for Twitter login page with options
login_options = Options()
login_options.add_argument("--start-maximized")
login_driver = webdriver.Chrome(options=login_options)

# Open Twitter login page
login_driver.get("https://twitter.com/login")

try:
    # Wait for the username field to be visible
    username_field = WebDriverWait(login_driver, 20).until(EC.visibility_of_element_located((By.NAME, 'session[username_or_email]')))
    username_field.send_keys(username)
    username_field.send_keys(Keys.ENTER)
    
    # Wait for the password field to be visible
    password_field = WebDriverWait(login_driver, 10).until(EC.visibility_of_element_located((By.NAME, 'session[password]')))
    password_field.send_keys(password)
    password_field.send_keys(Keys.ENTER)
    
    # Wait for the login process to complete
    WebDriverWait(login_driver, 20).until(EC.url_matches("https://twitter.com/home"))
    
    print("Login successful!")
except Exception as e:
    print("Failed to login:", str(e))
    login_driver.quit()
    exit(1)

# Now you're logged in, you can navigate to the target URL
target_url = "https://twitter.com/purethabang/followers"
login_driver.get(target_url)

# Find the element to scroll
wait = WebDriverWait(login_driver, 3)

try:
    # Wait until the "Load more" button is visible and clickable
    load_more_button = wait.until(EC.element_to_be_clickable((By.ID, "followers_you_follow-more")))
    
    # Scroll to the "Load more" button
    login_driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
    
    # Click on the "Load more" button
    load_more_button.click()
except TimeoutException:
    print("Timeout: Load more button not found or clickable.")

# Scrolling logic
prev_height = -1
max_scrolls = 1
scroll_count = 0

while scroll_count < max_scrolls:
    login_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)  # give some time for new results to load
    new_height = login_driver.execute_script("return document.body.scrollHeight")
    if new_height == prev_height:
        break
    prev_height = new_height
    scroll_count += 1

# Extract all elements with class "tweet-header-handle"
tweet_handles = login_driver.find_elements(By.CLASS_NAME, 'tweet-header-handle')

# Extract text from tweet handles, strip "@" symbol, and store them in a list
handle_list = [handle.text.lstrip('@') for handle in tweet_handles[:10]]

# Close the WebDriver for the login page
login_driver.quit()

# Call the function to scrape Twitter information for each handle
scrape_twitter_info(handle_list)