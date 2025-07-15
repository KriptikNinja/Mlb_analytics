#!/usr/bin/env python3
"""
Batch Statcast Processor
Automatically processes all Statcast files in the statcast_data directory
"""

import os
import glob
from statcast_processor import StatcastProcessor
import time

def process_all_statcast_files():
    """
    Process all CSV files in the statcast_data directory
    """
    print("🚀 Batch Statcast File Processor")
    print("=" * 50)
    
    # Find all CSV files in statcast_data directory
    csv_files = glob.glob("statcast_data/*.csv")
    
    if not csv_files:
        print("📂 No CSV files found in statcast_data directory")
        print("💡 Upload your Statcast files to statcast_data/ folder")
        return
    
    print(f"📊 Found {len(csv_files)} Statcast file(s) to process:")
    for file in csv_files:
        print(f"   • {os.path.basename(file)}")
    
    # Initialize processor
    processor = StatcastProcessor()
    
    # Process each file
    successful = 0
    failed = 0
    
    for file_path in csv_files:
        print(f"\n" + "="*60)
        print(f"🔄 Processing: {os.path.basename(file_path)}")
        print("="*60)
        
        try:
            success = processor.process_statcast_file(file_path)
            if success:
                successful += 1
                print(f"✅ Successfully processed {os.path.basename(file_path)}")
            else:
                failed += 1
                print(f"❌ Failed to process {os.path.basename(file_path)}")
        except Exception as e:
            failed += 1
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
        
        # Small delay between files
        time.sleep(1)
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"📊 BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed: {failed} files")
    print(f"📈 Total files: {len(csv_files)}")
    
    if successful > 0:
        print(f"\n🎯 Betting intelligence enhanced with {successful} Statcast file(s)!")
        print(f"🚀 Advanced metrics now available for enhanced edge detection")

def monitor_directory():
    """
    Monitor statcast_data directory for new files (future enhancement)
    """
    print("👀 Directory monitoring feature - coming soon!")
    print("   • Automatic detection of new uploads")
    print("   • Real-time processing of Statcast files") 
    print("   • Instant integration into betting system")

if __name__ == "__main__":
    process_all_statcast_files()