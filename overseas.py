import os
import csv
import re
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Global league mapping
LEAGUE_MAP = {
    'IND': 'IPL',
    'BAN': 'BPL',
    'SL': 'LPL',
    'PAK': 'PSL',
    'SA': 'SA20',
    'WI': 'CPL',
    'AUS': 'BBL',
    'ENG': 'T20 Blast/Hundred'
}

# Setup Chrome options once
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1920,1080')

# Function to check if a player is already in the CSV file
def is_player_in_csv(player_name, csv_file="players.csv"):
    if not os.path.exists(csv_file):
        return False
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        try:
            header = next(reader)  # Skip header
        except StopIteration:
            return False
        for row in reader:
            if row and row[0] == player_name:  # ✅ Check if row is not empty
                return True
    return False


# Function to scrape and write player data
def stats_taking(player):
    print(f"Processing player: {player}")
    data = {player: {}}
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        driver.get('https://stats.espncricinfo.com/ci/engine/stats/index.html')
        search_box = driver.find_element(By.NAME, "search")
        search_box.send_keys(player.strip())
        search_box.send_keys(Keys.RETURN)

        link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[starts-with(text(), 'Players and Officials')]"))
        )
        link.click()

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "gurusearch_player")))
        table = driver.find_element(By.ID, "gurusearch_player")

        rows = table.find_elements(By.XPATH, ".//table/tbody/tr[@valign='top']")
        max_matches = 0
        player_info = None

        for row in rows:
            try:
                match_links = row.find_elements(By.XPATH, ".//td[3]/a[contains(text(), 'Twenty20 matches player')]")
                for link in match_links:
                    text = link.find_element(By.XPATH, "./..").text
                    match = re.search(r"(\d+) matches", text)
                    if match:
                        matches = int(match.group(1))
                        if matches > max_matches:
                            max_matches = matches
                            nationality = row.find_elements(By.TAG_NAME, "td")[1].text
                            league = LEAGUE_MAP.get(nationality, "Overseas")
                            data[player] = {
                                "matches": matches,
                                "nationality": nationality,
                                "league": league
                            }
                            player_info = [player, league, nationality]
            except Exception as e:
                print(f"Error parsing row for {player}: {e}")
                continue

        if not player_info:
            print(f"No valid data for '{player}', defaulting to Overseas")
            player_info = [player, "Overseas", "Overseas"]

        # Write to CSV
        file_exists = os.path.exists("players.csv")
        with open("players.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["player", "league", "nationality"])
            writer.writerow(player_info)
        print(f"Processed: {player} -> {player_info[1]}")

    except (NoSuchElementException, TimeoutException) as e:
        print(f"Failed to fetch data for {player}: {e}")
    finally:
        driver.quit()


def process_json_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as json_file:
                    try:
                        data = json.load(json_file)
                        if data.get("info", {}).get("gender") == "female":
                            continue
                        players = data.get("info", {}).get("players", {})
                        for team_players in players.values():
                            for player in team_players:
                                if not is_player_in_csv(player):
                                    stats_taking(player)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding {file_path}: {e}")

# Call the main function
process_json_files("/Users/jatin/Documents/python/multipremierleague")
