# Statcast Data Directory

This folder is for uploading your Statcast CSV files to enhance the betting analytics system.

## How to Use

### Option 1: Individual CSV Files
1. **Upload Files**: Drop your Statcast CSV files directly into this folder
2. **Run Processing**: Execute `python batch_statcast_processor.py`

### Option 2: Zip Archives (Recommended for Multiple Files)
1. **Upload Zip**: Drop your zip file containing multiple Statcast CSV files into this folder
2. **Auto-Extract & Process**: Execute `python zip_statcast_processor.py`
3. **Automatic Cleanup**: Processed zips moved to `processed_zips/` folder

### Features
- **Duplicate Protection**: Built-in safeguards prevent data duplication
- **Quality Filtering**: Automatically removes records with missing dates/players
- **Overwrite Safe**: Database merge prevents conflicts with existing data
- **Enhanced Analytics**: Adds advanced metrics like exit velocity, launch angle, hard-hit rate

## File Requirements

Your Statcast files should contain these key columns:
- `player_name` - Player identification
- `game_date` - Game date
- `events` - Plate appearance outcomes
- `launch_speed` - Exit velocity (optional but valuable)
- `launch_angle` - Launch angle (optional but valuable)
- `hit_distance_sc` - Hit distance (optional)

## Processing Status

✅ **statcast_673357_1752329574688.csv** - Successfully processed
- 4,954 pitch records → 810 game records
- 358 unique players with advanced metrics
- Date range: 2023-03-30 to 2025-06-24

## Data Integration

Your Statcast data seamlessly integrates with MLB Stats API data:
- No duplicates created (database constraints protect integrity)
- Advanced metrics enhance basic statistics
- Richer betting intelligence and edge detection
- Background collection continues for comprehensive coverage

## Next Steps

Upload additional Statcast files here to expand the dataset further. Each file will be automatically processed and integrated into the historical database for enhanced betting analytics.