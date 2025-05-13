from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import json
import time
from pathlib import Path
import logging
import traceback
import sys

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_driver():
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        driver = webdriver.Chrome(options=options)
        logging.info("Chrome driver initialized successfully")
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize Chrome driver: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        raise

def find_all_cards(driver):
    """Recursively find all card elements in the page."""
    try:
        # Look for any element with data-card attribute
        card_elements = driver.find_elements(By.CSS_SELECTOR, "[data-card]")
        logging.info(f"Found {len(card_elements)} card elements")
        return card_elements
    except Exception as e:
        logging.error("Error finding card elements:", exc_info=True)
        return []

def extract_card_data(elements):
    cards = []
    for element in elements:
        try:
            # Get the parent container to determine deck type
            parent = element.find_element(By.XPATH, "./ancestor::div[contains(@id, '_deck')]")
            deck_type = parent.get_attribute('id').replace('_deck', '')
            
            card_data = {
                'name': element.get_attribute('data-cardname'),
                'id': element.get_attribute('data-card'),
                'type': element.get_attribute('data-cardtype'),
                'image_url': element.get_attribute('src'),
                'deck_type': deck_type
            }
            cards.append(card_data)
        except StaleElementReferenceException:
            logging.warning("Stale element encountered, skipping...", exc_info=True)
            continue
        except Exception as e:
            logging.warning(f"Error extracting card data: {str(e)}", exc_info=True)
            continue
    return cards

def get_deck_cards(driver, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting to load {url} (attempt {attempt + 1}/{max_retries})")
            driver.get(url)
            
            # Wait for the page to load completely
            logging.info("Waiting for body element...")
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for any card elements to be present
            logging.info("Waiting for card elements...")
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-card]"))
            )
            
            # Find all cards
            all_cards = find_all_cards(driver)
            if not all_cards:
                logging.error("No cards found on the page")
                raise Exception("No cards found")
            
            # Extract and organize cards by deck type
            cards_data = extract_card_data(all_cards)
            
            # Organize cards into their respective decks
            deck_data = {
                "main_deck": [],
                "extra_deck": [],
                "side_deck": []
            }
            
            for card in cards_data:
                deck_type = card.pop('deck_type', 'main')  # Default to main if not specified
                deck_data[f"{deck_type}_deck"].append(card)
            
            logging.info(f"Successfully scraped deck from {url}")
            return deck_data
            
        except TimeoutException as e:
            logging.error(f"Timeout while loading {url} (attempt {attempt + 1}/{max_retries})")
            logging.error("Timeout details:", exc_info=True)
            if attempt < max_retries - 1:
                logging.info(f"Waiting 5 seconds before retry {attempt + 2}")
                time.sleep(5)
                continue
            return None
        except Exception as e:
            logging.error(f"Error processing {url}")
            logging.error("Error details:", exc_info=True)
            if attempt < max_retries - 1:
                logging.info(f"Waiting 5 seconds before retry {attempt + 2}")
                time.sleep(5)
                continue
            return None
    
    return None

def main():
    try:
        # Read the meta_decks.json file
        logging.info("Reading meta_decks.json file")
        with open('meta_decks.json', 'r') as f:
            decks = json.load(f)
        
        driver = setup_driver()
        results = []
        
        try:
            for deck in decks:
                logging.info(f"Processing {deck['archtype']} - {deck['url']}")
                deck_data = get_deck_cards(driver, deck['url'])
                
                if deck_data:
                    results.append({
                        "archtype": deck['archtype'],
                        "url": deck['url'],
                        "cards": deck_data
                    })
                
                # Add a longer delay to avoid overwhelming the server
                time.sleep(5)
        
        finally:
            logging.info("Closing Chrome driver")
            driver.quit()
        
        # Save results to a new JSON file
        output_file = 'deck_cards.json'
        logging.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Scraping completed. Results saved to {output_file}")
        
    except Exception as e:
        logging.error("Fatal error in main process")
        logging.error("Error details:", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Script failed with error")
        logging.error("Error details:", exc_info=True)
        sys.exit(1) 