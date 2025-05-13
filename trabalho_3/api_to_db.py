import requests
import sqlite3

API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
DB_NAME = "cards.db"


def fetch_all_cards():
    response = requests.get(API_URL)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")
    cards = response.json().get("data", [])
    return cards


def create_tables(cursor):
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT,
        frameType TEXT,
        desc TEXT,
        atk INTEGER,
        def INTEGER,
        level INTEGER,
        race TEXT,
        attribute TEXT
    );

    CREATE TABLE IF NOT EXISTS card_sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER,
        set_name TEXT,
        set_code TEXT,
        set_rarity TEXT,
        set_rarity_code TEXT,
        set_price REAL,
        FOREIGN KEY (card_id) REFERENCES cards(id)
    );

    CREATE TABLE IF NOT EXISTS card_images (
        id INTEGER PRIMARY KEY,
        card_id INTEGER,
        image_url TEXT,
        image_url_small TEXT,
        image_url_cropped TEXT,
        FOREIGN KEY (card_id) REFERENCES cards(id)
    );

    CREATE TABLE IF NOT EXISTS card_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER,
        cardmarket_price REAL,
        tcgplayer_price REAL,
        ebay_price REAL,
        amazon_price REAL,
        coolstuffinc_price REAL,
        FOREIGN KEY (card_id) REFERENCES cards(id)
    );
    ''')


def insert_card(cursor, card):
    cursor.execute('''
        INSERT OR IGNORE INTO cards (id, name, type, frameType, desc, atk, def, level, race, attribute)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        card["id"],
        card["name"],
        card.get("type"),
        card.get("frameType"),
        card.get("desc"),
        card.get("atk"),
        card.get("def"),
        card.get("level"),
        card.get("race"),
        card.get("attribute")
    ))

    for s in card.get("card_sets", []):
        cursor.execute('''
        INSERT INTO card_sets (card_id, set_name, set_code, set_rarity, set_rarity_code, set_price)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            card["id"],
            s["set_name"],
            s["set_code"],
            s["set_rarity"],
            s["set_rarity_code"],
            float(s["set_price"]) if s["set_price"] else 0.0
        ))

    for img in card.get("card_images", []):
        cursor.execute('''
        INSERT OR IGNORE INTO card_images (id, card_id, image_url, image_url_small, image_url_cropped)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            img["id"],
            card["id"],
            img["image_url"],
            img["image_url_small"],
            img["image_url_cropped"]
        ))

    for price in card.get("card_prices", []):
        cursor.execute('''
        INSERT INTO card_prices (
            card_id, cardmarket_price, tcgplayer_price,
            ebay_price, amazon_price, coolstuffinc_price
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            card["id"],
            float(price.get("cardmarket_price", 0.0)),
            float(price.get("tcgplayer_price", 0.0)),
            float(price.get("ebay_price", 0.0)),
            float(price.get("amazon_price", 0.0)),
            float(price.get("coolstuffinc_price", 0.0))
        ))


def main():
    cards = fetch_all_cards()

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    create_tables(cursor)

    for i, card in enumerate(cards, 1):
        insert_card(cursor, card)
        if i % 500 == 0:
            print(f"  ...{i} cards inserted")
            conn.commit()  # Commit in batches for performance

    conn.commit()
    conn.close()
    print(f"All {len(cards)} cards imported into '{DB_NAME}'.")


if __name__ == "__main__":
    main()
