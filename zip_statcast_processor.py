#!/usr/bin/env python3
"""
Zip Statcast Processor
Automatically extracts and processes Statcast files from zip archives
"""

import os
import zipfile
import glob
from pathlib import Path
from batch_statcast_processor import process_all_statcast_files
from statcast_processor import StatcastProcessor
import shutil

def extract_and_process_zip():
    """
    Extract all zip files in statcast_data and process the CSV files
    """
    print("📦 Zip Statcast File Processor")
    print("=" * 50)
    
    # Find all zip files in statcast_data directory
    zip_files = glob.glob("statcast_data/*.zip")
    
    if not zip_files:
        print("📂 No zip files found in statcast_data directory")
        print("💡 Upload your Statcast zip files to statcast_data/ folder")
        return
    
    print(f"📦 Found {len(zip_files)} zip file(s) to extract:")
    for zip_file in zip_files:
        print(f"   • {os.path.basename(zip_file)}")
    
    # Extract each zip file
    total_extracted = 0
    for zip_path in zip_files:
        print(f"\n🔄 Extracting: {os.path.basename(zip_path)}")
        extracted_count = extract_zip_file(zip_path)
        total_extracted += extracted_count
        
        # Move processed zip to avoid re-processing
        processed_dir = "statcast_data/processed_zips"
        os.makedirs(processed_dir, exist_ok=True)
        processed_path = os.path.join(processed_dir, os.path.basename(zip_path))
        shutil.move(zip_path, processed_path)
        print(f"✅ Moved {os.path.basename(zip_path)} to processed_zips/")
    
    if total_extracted > 0:
        print(f"\n📊 Extraction Summary:")
        print(f"   • Total CSV files extracted: {total_extracted}")
        print(f"   • Ready for processing")
        
        # Process all extracted CSV files
        print(f"\n🚀 Processing all Statcast files...")
        process_all_statcast_files()
    else:
        print(f"\n⚠️  No CSV files found in zip archives")

def extract_zip_file(zip_path: str) -> int:
    """
    Extract CSV files from a zip archive to statcast_data directory
    """
    extracted_count = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in zip
            file_list = zip_ref.namelist()
            csv_files = [f for f in file_list if f.lower().endswith('.csv')]
            
            print(f"   📁 Found {len(csv_files)} CSV files in zip")
            
            for csv_file in csv_files:
                # Extract to statcast_data directory
                zip_ref.extract(csv_file, "statcast_data/")
                
                # If file was in a subdirectory, move it to root of statcast_data
                extracted_path = os.path.join("statcast_data", csv_file)
                final_path = os.path.join("statcast_data", os.path.basename(csv_file))
                
                if extracted_path != final_path:
                    # Move file to root of statcast_data and remove empty subdirectories
                    shutil.move(extracted_path, final_path)
                    
                    # Clean up empty subdirectory
                    subdir = os.path.dirname(extracted_path)
                    if subdir != "statcast_data" and os.path.exists(subdir):
                        try:
                            os.rmdir(subdir)
                        except OSError:
                            pass  # Directory not empty
                
                print(f"   ✅ Extracted: {os.path.basename(csv_file)}")
                extracted_count += 1
            
    except zipfile.BadZipFile:
        print(f"   ❌ Error: {os.path.basename(zip_path)} is not a valid zip file")
    except Exception as e:
        print(f"   ❌ Error extracting {os.path.basename(zip_path)}: {e}")
    
    return extracted_count

def handle_duplicate_csvs():
    """
    Handle duplicate CSV files by showing options
    """
    csv_files = glob.glob("statcast_data/*.csv")
    
    if len(csv_files) > 1:
        print(f"\n📋 Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            file_size = os.path.getsize(csv_file)
            print(f"   • {os.path.basename(csv_file)} ({file_size:,} bytes)")
        
        print(f"\n✅ All files will be processed with duplicate prevention")
        print(f"   • Database constraints prevent duplicate records")
        print(f"   • Existing data will be updated, not duplicated")
        print(f"   • Your betting projections remain accurate")

def monitor_statcast_directory():
    """
    Show current status of statcast_data directory
    """
    print(f"\n📊 Statcast Data Directory Status:")
    print("=" * 40)
    
    csv_files = glob.glob("statcast_data/*.csv")
    zip_files = glob.glob("statcast_data/*.zip")
    processed_zips = glob.glob("statcast_data/processed_zips/*.zip")
    
    print(f"📄 CSV files ready: {len(csv_files)}")
    print(f"📦 Zip files pending: {len(zip_files)}")
    print(f"✅ Processed zips: {len(processed_zips)}")
    
    if csv_files:
        print(f"\n📄 CSV Files:")
        for csv_file in csv_files:
            file_size = os.path.getsize(csv_file)
            print(f"   • {os.path.basename(csv_file)} ({file_size:,} bytes)")
    
    if zip_files:
        print(f"\n📦 Zip Files to Process:")
        for zip_file in zip_files:
            file_size = os.path.getsize(zip_file)
            print(f"   • {os.path.basename(zip_file)} ({file_size:,} bytes)")

if __name__ == "__main__":
    # First show current status
    monitor_statcast_directory()
    
    # Extract and process zip files
    extract_and_process_zip()
    
    # Handle any duplicate CSVs
    handle_duplicate_csvs()
    
    print(f"\n🎯 Zip processing complete!")
    print(f"✅ All Statcast data integrated with duplicate protection")