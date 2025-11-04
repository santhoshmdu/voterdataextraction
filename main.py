#!/usr/bin/env python3
"""
PDF OCR Data Extractor - v2 (Accuracy Enhanced)

Usage: python main_accuracy_v2.py TN_Tenkasi path/to/your.pdf --rpm 1500

Features:
- **ACCURACY**: Requests structured JSON from API to eliminate CSV parsing errors.
- **ACCURACY**: Uses OpenCV for advanced image pre-processing (binarization) to improve raw OCR quality.
- **ACCURACY**: Post-processes data to forward-fill missing section headers.
- Asynchronous producer-consumer pattern using asyncio and a queue.
- Decoupled PDF-to-image conversion and API processing for efficiency.
- High-throughput API rate limiter to manage 1500-2000+ RPM safely.
- Parallelized, RAM-only image optimization.
- Final data is sorted by page number for logical output.
- Robust error handling and logging.
- Loads Gemini API key from .env file securely.
- Output file named: <Location>_<PDF basename>_processed.xlsx
"""

import os
import sys
import time
import io
import json
import argparse
import pandas as pd
import asyncio
import collections
from pdf2image import convert_from_bytes
import google.generativeai as genai
from PIL import Image
import re
import csv
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor


# Define Poppler path relative to your project folder
poppler_path = os.path.join(os.getcwd(), "poppler-25.07.0", "Library", "bin")

# Add Poppler path to system PATH at runtime
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + poppler_path
# --- NEW DEPENDENCY: OpenCV for image pre-processing ---
try:
    import cv2
    import numpy as np
except ImportError:
    print("❌ Missing 'opencv-python-headless'. Install with: pip install opencv-python-headless")
    sys.exit(1)

# dotenv for environment variables
try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ Missing 'python-dotenv'. Install with: pip install python-dotenv")
    sys.exit(1)

load_dotenv()  # Load environment variables from .env

# --- ACCURACY ENHANCEMENT: Image Processing with OpenCV ---
def process_image_for_accuracy(image_obj, max_dimension, quality):
    """
    Optimizes a PIL image for OCR accuracy using OpenCV (binarization)
    before resizing and compressing it.
    """
    # Convert PIL Image to OpenCV format (from RGB to BGR)
    open_cv_image = cv2.cvtColor(np.array(image_obj), cv2.COLOR_RGB2BGR)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # 2. Binarization (Adaptive Thresholding) - This sharpens text and removes noise
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert the processed OpenCV image back to a PIL Image
    processed_pil_image = Image.fromarray(binary)
    
    # Resize while maintaining aspect ratio
    if processed_pil_image.width > max_dimension or processed_pil_image.height > max_dimension:
        processed_pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    # Save to an in-memory bytes buffer with specified quality
    img_byte_arr = io.BytesIO()
    processed_pil_image.save(
        img_byte_arr,
        format='JPEG',
        quality=quality,
        optimize=True
    )
    return img_byte_arr.getvalue()


class APIRateLimiter:
    """
    An asyncio-compatible rate limiter using a sliding window algorithm (via a deque).
    This ensures the number of requests in any 60-second window does not exceed the limit,
    making it suitable for high-throughput, concurrent environments.
    """
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            self.max_requests = float('inf')
        else:
            self.max_requests = requests_per_minute
        self.request_timestamps = collections.deque()
        self.lock = asyncio.Lock()
        if self.max_requests != float('inf'):
             print(f"🚦 Rate limiter initialized to {self.max_requests} requests/minute.")

    async def wait(self):
        if self.max_requests == float('inf'):
            return
        async with self.lock:
            current_time = time.monotonic()
            while self.request_timestamps and self.request_timestamps[0] < current_time - 60:
                self.request_timestamps.popleft()
            if len(self.request_timestamps) >= self.max_requests:
                time_to_wait = self.request_timestamps[0] - (current_time - 60)
                if time_to_wait > 0:
                    await asyncio.sleep(time_to_wait)
            self.request_timestamps.append(time.monotonic())


class AsyncPDFExtractor:
    def __init__(self, api_key, num_workers, requests_per_minute):
        self.API_KEY = api_key
        self.num_workers = num_workers
        self.rate_limiter = APIRateLimiter(requests_per_minute)
        try:
            genai.configure(api_key=self.API_KEY)
            system_instruction = "You are a professional electoral roll OCR specialist. Your sole job is to extract data accurately from PDF page images into a structured JSON format."
            self.model = genai.GenerativeModel(
                "gemini-2.5-pro", # Using 1.5 Flash for better JSON adherence
                system_instruction=system_instruction
            )
            print(f"✅ Gemini AI initialized successfully. Using {self.num_workers} concurrent workers.")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini AI: {e}")
            sys.exit(1)

        # Expected columns for the final DataFrame
        self.final_columns = [
            'Section No and Name', 'S.No', 'VoterID', 'Name',
            'Father/Husband Name', 'House Number', 'Age', 'Gender',
            'Photo Availability', 'Status'
        ]
        self.results = []
        self.failed_info = []

    def log_message(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    async def producer(self, pdf_path, queue, dpi, max_dimension, quality, executor):
        self.log_message("🚀 Starting PDF-to-image conversion (Producer)...")
        loop = asyncio.get_running_loop()
        try:
            with open(pdf_path, 'rb') as file:
                pdf_bytes = file.read()

            images = await asyncio.to_thread(convert_from_bytes, pdf_bytes, dpi=dpi)

            total_pages = len(images)
            self.log_message(f"🖼️ PDF converted to {total_pages} images. Populating queue with accuracy-optimized images...")

            pages_to_process = images[2:-1]
            
            tasks = [
                loop.run_in_executor(
                    executor, process_image_for_accuracy, page_obj, max_dimension, quality
                ) for page_obj in pages_to_process
            ]
            
            for i, task in enumerate(asyncio.as_completed(tasks)):
                page_idx = i + 2
                compressed_bytes = await task
                await queue.put((page_idx, compressed_bytes))

            self.log_message("✅ Producer finished. All optimized pages are in the queue.")
            return total_pages
        except Exception as e:
            self.log_message(f"❌ CRITICAL ERROR in producer: {e}")
            for _ in range(self.num_workers):
                await queue.put(None)
            return 0

    async def consumer(self, queue, complete_prompt):
        while True:
            item = await queue.get()
            if item is None:
                break
            page_idx, img_bytes = item
            try:
                df, fail_info = await self.async_process_single_page(page_idx, img_bytes, complete_prompt)
                if df is not None:
                    self.results.append(df)
                elif fail_info:
                    self.failed_info.append(fail_info)
            except Exception as e:
                self.failed_info.append({'page_num': page_idx + 1, 'raw_response': '', 'error': f"Consumer-level exception: {e}"})
            finally:
                queue.task_done()

    async def async_process_single_page(self, page_idx, img_bytes, complete_prompt):
        page_num = page_idx + 1
        try:
            max_api_retries = 3
            api_response = None
            for attempt in range(max_api_retries):
                try:
                    await self.rate_limiter.wait()
                    api_response = await self.model.generate_content_async([
                        complete_prompt, {"mime_type": "image/jpeg", "data": img_bytes}
                    ])
                    break
                except Exception as api_error:
                    self.log_message(f"⚠️ API call failed for page {page_num} (Attempt {attempt+1}/{max_api_retries}): {api_error}")
                    if attempt == max_api_retries - 1:
                        raise api_error
                    await asyncio.sleep(2)

            if api_response is None: raise Exception("API response is None after retries")

            raw_response = api_response.text.strip()
            
            # --- NEW: Call the JSON parser ---
            df = self.parse_json_response(raw_response, page_num)

            if df is not None and not df.empty:
                df['Page_Number'] = page_num
                df['Processing_Timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                df = self.validate_and_fix_dataframe(df, page_num)
                self.log_message(f"✅ Page {page_num}: Successfully processed and parsed.")
                return (df, None)
            else:
                self.log_message(f"❌ Page {page_num}: Failed to parse valid data from JSON response.")
                return (None, {'page_num': page_num, 'raw_response': raw_response, 'error': 'JSON parsing failed or returned empty data'})

        except Exception as page_error:
            self.log_message(f"❌ Page {page_num}: Critical error - {page_error}")
            return (None, {'page_num': page_num, 'raw_response': getattr(api_response, 'text', ''), 'error': str(page_error)})

    async def process_pdf(self, pdf_path, location_prefix, dpi, section_mode, rpm, max_dimension, quality):
        start_time = time.time()
        page_queue = asyncio.Queue(maxsize=self.num_workers * 2)

        # --- ACCURACY: New prompt requesting JSON output ---
        complete_prompt = """Extract section info and all voter data from the image as a single, valid JSON object.

CRITICAL REQUIREMENTS:
1.  The root of the object must contain two keys: "SECTION_INFO" and "voters".
2.  The value for "SECTION_INFO" must be a string containing the full section name from the top of the page.
3.  The value for "voters" must be a LIST of voter objects.
4.  Each voter object in the list MUST contain these exact keys: "S_No", "VoterID", "Name", "Father_Husband_Name", "House_Number", "Age", "Gender", "Photo_Availability", "Status".

FORMATTING RULES:
- The entire output MUST be a single JSON object. Do not wrap it in markdown.
- If a value is not found, use an empty string "" or null.
- "Age" must be an integer.
- "Status": "Active" or "Deleted" (if the entry is struck through).
- "Photo_Availability": "Yes" or "No".
- "Gender": "Male" or "Female".

EXAMPLE JSON OUTPUT:
{
  "SECTION_INFO": "1-Viswanathaperi-1 (R.V), Viswanathaperi (P), Ward no.9 gandhi colony st",
  "voters": [
    {
      "S_No": 1,
      "VoterID": "ABC123",
      "Name": "John Smith",
      "Father_Husband_Name": "Father Name",
      "House_Number": "123",
      "Age": 25,
      "Gender": "Male",
      "Photo_Availability": "Yes",
      "Status": "Active"
    },
  ]
}
"""
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            consumer_tasks = [asyncio.create_task(self.consumer(page_queue, complete_prompt)) for _ in range(self.num_workers)]
            producer_task = asyncio.create_task(self.producer(pdf_path, page_queue, dpi, max_dimension, quality, executor))
            total_pages = await producer_task

            if total_pages == 0:
                self.log_message("❌ Producer failed. Shutting down.")
                [task.cancel() for task in consumer_tasks]
                return False

            await page_queue.join()
            for _ in range(self.num_workers):
                await page_queue.put(None)
            await asyncio.gather(*consumer_tasks, return_exceptions=True)

        if self.results:
            combined_df = pd.concat(self.results, ignore_index=True)
            if 'Page_Number' in combined_df.columns:
                self.log_message("Sorting all extracted data by page number...")
                combined_df = combined_df.sort_values(by=['Page_Number', 'S.No']).reset_index(drop=True)

            # --- ACCURACY: Forward fill for missing section headers ---
            self.log_message("Fixing missing section headers...")
            default_section_pattern = r"SECTION_INFO_MISSING_PAGE_\d+"
            combined_df['Section No and Name'] = combined_df['Section No and Name'].replace(
                to_replace=default_section_pattern, value=pd.NA, regex=True
            )
            combined_df['Section No and Name'].ffill(inplace=True)

            # Reorder columns to the desired final format
            final_df = combined_df.reindex(columns=self.final_columns + ['Page_Number', 'Processing_Timestamp'])


            folder_path = './output'
            os.makedirs(folder_path, exist_ok=True)
            pdf_basename = os.path.basename(pdf_path)
            file_stem = os.path.splitext(pdf_basename)[0]
            excel_path = os.path.join(folder_path, f"{location_prefix}_{file_stem}_processed.xlsx") #Excel name
            csv_path = os.path.join(folder_path, f"{location_prefix}_{file_stem}_processed.csv")
            final_df.to_excel(excel_path, index=False, engine='openpyxl')
            final_df.to_csv(csv_path, index=False)
            
            self.log_message("🎉 ========== COMPLETE PROCESSING SUCCESS ==========")
            self.log_message(f"📊 Records extracted: {len(final_df)}")
            self.log_message(f"✅ Successful pages: {len(self.results)}")
            self.log_message(f"❌ Failed pages: {len(self.failed_info)}")
            self.log_message(f"⏱️ Processing time: {(time.time() - start_time):.1f} seconds")
            self.log_message(f"📄 Excel: {excel_path}")
            return True
        else:
            self.log_message("❌ No data could be extracted from any page")
            return False

    def parse_json_response(self, raw_text, page_num):
        try:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not match:
                self.log_message(f"⚠️ Page {page_num}: No JSON object found in response.")
                return None
            
            json_str = match.group(0)
            data = json.loads(json_str)
            
            section_info = data.get("SECTION_INFO", f"SECTION_INFO_MISSING_PAGE_{page_num}")
            voters_list = data.get("voters", [])
            
            if not voters_list or not isinstance(voters_list, list):
                return None

            df = pd.DataFrame(voters_list)
            df['Section No and Name'] = section_info
            return df
        except (json.JSONDecodeError, TypeError) as e:
            self.log_message(f"❌ Page {page_num}: Failed to parse JSON. Error: {e}")
            return None

    def validate_and_fix_dataframe(self, df, page_num):
        if df is None or df.empty: return None
        
        # Rename columns to match the final expected format
        column_mapping = {
            'S_No': 'S.No', 'Father_Husband_Name': 'Father/Husband Name',
            'House_Number': 'House Number', 'Photo_Availability': 'Photo Availability'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Ensure all expected columns exist
        for col in self.final_columns:
            if col not in df.columns:
                df[col] = ''
        
        df = df.fillna('').astype(str)
        df = df[df['Name'].str.strip() != '']
        if df.empty: return None

        df['S.No'] = pd.to_numeric(df['S.No'], errors='coerce').fillna(0).astype(int)
        df['Status'] = df['Status'].apply(lambda x: 'Deleted' if 'delet' in str(x).lower() else 'Active')
        df['Photo Availability'] = df['Photo Availability'].apply(lambda x: 'Yes' if str(x).lower() in ['yes', 'y', 'true', '1'] else 'No')
        df['Gender'] = df['Gender'].apply(lambda x: str(x).capitalize())
        
        return df

async def main():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found. Please set it in .env file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='PDF OCR Data Extractor - v2 (Accuracy Enhanced)')
    parser.add_argument('location', help='Location prefix for output filename')
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for image conversion (default: 200)')
    parser.add_argument('--workers', type=int, default=min(cpu_count() * 4, 20), help='Number of concurrent API workers')
    parser.add_argument('--rpm', type=int, default=1500, help='Max requests per minute for the Gemini API')
    parser.add_argument('--jpeg-quality', type=int, default=80, help='JPEG quality for images (1-95, default: 80)')
    parser.add_argument('--max-dimension', type=int, default=1440, help='Maximum width/height for images (default: 1600)')
    
    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"❌ File not found: {args.pdf_file}")
        return

    print("📄 Starting Accuracy Enhanced PDF OCR Extractor v2")
    print(f"🔧 Settings: DPI={args.dpi}, Workers={args.workers}, RPM Limit={args.rpm}, JPEG Quality={args.jpeg_quality}, Max Dimension={args.max_dimension}")
    
    extractor = AsyncPDFExtractor(api_key, args.workers, args.rpm)
    success = await extractor.process_pdf(
        args.pdf_file, args.location, args.dpi, 'json_mode', 
        args.rpm, args.max_dimension, args.jpeg_quality
    )

    if success:
        print("\n✅ Processing completed successfully!")
    else:
        print("\n❌ Processing failed or produced no data!")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())