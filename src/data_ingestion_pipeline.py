import requests
import os
import json
import time

def download_statsbomb_free_data(target_dir: str = "data/raw/statsbomb_free"):
    """
    Automated downloader for StatsBomb Open Data.
    Fetches all available competitions, creates structured folders by League/Season,
    and downloads match event JSONs with local caching enabled.
    """
    
    print('Fetching Master competitions index from StatsBomb...')
    competitions_url = "https://raw.githubusercontent.com/hudl/open-data/refs/heads/master/data/competitions.json"
    comp_response = requests.get(competitions_url)

    if comp_response.status_code != 200:
        print(f"Failed to fetch data. Status code: {comp_response.status_code}")
        return

    competitions = comp_response.json()
    print(f"Master index loaded! Found {len(competitions)} competition-season pairs.")

    total_downloaded = 0
    total_skipped = 0
    seasons_num = 0

    for comp in competitions:

        seasons_num += 1

        competition_id = comp['competition_id']
        season_id = comp['season_id']

        competition_name = comp['competition_name'].replace("/", "-")
        season_name = comp['season_name'].replace("/", "-")

        season_dir = os.path.join(target_dir, competition_name, season_name)
        os.makedirs(season_dir, exist_ok=True)

        matches_url = f"https://raw.githubusercontent.com/hudl/open-data/refs/heads/master/data/matches/{competition_id}/{season_id}.json"
        matches_response = requests.get(matches_url)

        if matches_response.status_code != 200:
            print(f"Could not fetch match list for {competition_name} ({season_name})")
            continue

        matches = matches_response.json()
        print(f'\nProcessing {competition_name} ({season_name}) - {len(matches)} matches found.')

        for match in matches:
            match_id = match['match_id']
            
            match_path = os.path.join(season_dir, f"{match_id}.json")
            
            if os.path.exists(match_path):
                total_skipped += 1
                continue

            event_url = f"https://raw.githubusercontent.com/hudl/open-data/refs/heads/master/data/events/{match_id}.json"
            event_response = requests.get(event_url)

            if event_response.status_code == 200:
                with open(match_path, 'w', encoding = 'utf-8') as f:
                    json.dump(event_response.json(), f)
                total_downloaded += 1
                time.sleep(0.05)
            else:
                print(f"Could not download event with the id: {match_id}")


    print("\n" + "="*50)
    print("AUTOMATED DATA INGESTION FROM STATSBOM COMPLETE!")
    print(f"Downloaded new match files: {total_downloaded}")
    print(f"Skipped existing match files: {total_skipped}")
    print(f"Total seasons processed: {seasons_num}")
    print("="*50)


if __name__ == "__main__":
    download_statsbomb_free_data()

            

