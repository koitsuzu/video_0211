import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from mistralai import Mistral

# 載入環境變數
load_dotenv()

def extract_audio(video_path: Path, audio_path: Path):
    """從影片提取音訊並儲存為 mp3"""
    print(f"正在從 {video_path.name} 提取音訊...")
    video = VideoFileClip(str(video_path))
    video.audio.write_audiofile(str(audio_path), logger=None)
    video.close()
    print(f"音訊已提取至 {audio_path}")

def transcribe_with_mistral(client: Mistral, audio_path: Path):
    """呼叫 Mistral API 進行轉錄，取得帶時間軸的 segment"""
    print(f"正在呼叫 Mistral API 進行轉錄...")
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.complete(
            model="voxtral-mini-latest",
            file=(audio_path.name, f.read()),
            timestamp_granularities=["segment"]
        )
    return response

def process_and_summarize(client: Mistral, transcription_response):
    """
    使用 Mistral Chat API 對逐字稿進行：
    1. 翻譯為繁體中文
    2. 產生內容摘要
    """
    print("正在翻譯逐字稿並產生摘要...")
    
    # 將 segments 轉為一段純文字以利翻譯與總結，保留 ID 以便對應
    segments = transcription_response.segments
    text_to_process = "\n".join([f"[{s.start}-{s.end}] {s.text}" for s in segments])
    
    prompt = f"""
你是一個專業的影音逐字稿翻譯與總結助手。
請將以下帶有時間軸的逐字稿內容翻譯為「繁體中文」，並提供完整的內容摘要。

### 原始逐字稿內容：
{text_to_process}

### 要求：
1. 輸出格式必須為 JSON。
2. JSON 結構如下：
{{
  "summary": "這裡填寫整體的繁體中文摘要",
  "translated_segments": [
    {{
      "start": 0.0,
      "end": 2.5,
      "text": "這裡填寫翻譯後的繁體中文內容"
    }}
  ]
}}
3. 務必保持原有的 start 與 end 時間數值不變。
4. 摘要必須精煉且包含重點。
5. 請只返回 JSON 內容。
"""

    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(chat_response.choices[0].message.content)

def main():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("錯誤：請在 .env 檔案中設定 MISTRAL_API_KEY")
        return

    client = Mistral(api_key=api_key)
    
    # 路徑設定
    video_dir = Path("Video")
    output_dir = Path("output")
    temp_dir = Path("temp_audio")
    
    output_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    # 支援的影片格式
    video_extensions = [".mp4", ".mkv", ".mov", ".avi"]
    
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    if not videos:
        print(f"在 {video_dir} 目錄下找不到影片檔案。")
        return

    for video_path in videos:
        print(f"\n--- 開始處理影片: {video_path.name} ---")
        
        # 1. 提取音訊
        audio_path = temp_dir / f"{video_path.stem}.mp3"
        extract_audio(video_path, audio_path)
        
        try:
            # 2. 轉錄
            transcription = transcribe_with_mistral(client, audio_path)
            
            # 3. 翻譯與摘要
            final_result = process_and_summarize(client, transcription)
            
            # 注入檔名資訊
            final_result["file_name"] = video_path.name
            
            # 4. 儲存結果
            output_file = output_dir / f"{video_path.stem}_transcription.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            print(f"完成！結果已儲存至: {output_file}")
            
        except Exception as e:
            print(f"處理影片時發生錯誤: {e}")
        
        finally:
            # 清理暫存音訊
            if audio_path.exists():
                audio_path.unlink()

if __name__ == "__main__":
    main()
